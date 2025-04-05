//! This example showcases an interactive version of the Game of Life, invented
//! by John Conway. It leverages a `Canvas` together with other widgets.
mod preset;

use grid::Grid;
use preset::Preset;

use iced::time;
use iced::widget::{button, checkbox, column, container, pick_list, row, slider, text};
use iced::{Center, Element, Fill, Subscription, Task, Theme};
use std::time::Duration;

use rand::Rng;

use std::collections::VecDeque;

pub fn main() -> iced::Result {
    tracing_subscriber::fmt::init();

    iced::application("Samuel Cusack's NEA", LifeSim::update, LifeSim::view)
        .subscription(LifeSim::subscription)
        .theme(|_| Theme::Dark)
        .antialiasing(true)
        .centered()
        .run()
}

struct LifeSim {
    grid: Grid,
    is_playing: bool,
    queued_ticks: usize,
    speed: usize,
    next_speed: Option<usize>,
    version: usize,
}

#[derive(Debug, Clone)]
enum Message {
    Grid(grid::Message, usize),
    Tick,
    TogglePlayback,
    ToggleGrid(bool),
    Next,
    Clear,
    SpeedChanged(f32),
    PresetPicked(Preset),
}

impl LifeSim {
    fn new() -> Self {
        Self {
            grid: Grid::default(),
            is_playing: false,
            queued_ticks: 0,
            speed: 5,
            next_speed: None,
            version: 0,
        }
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Grid(message, version) => {
                if version == self.version {
                    self.grid.update(message);
                }
            }
            Message::Tick | Message::Next => {
                self.queued_ticks = (self.queued_ticks + 1).min(self.speed);

                if let Some(task) = self.grid.tick(self.queued_ticks) {
                    if let Some(speed) = self.next_speed.take() {
                        self.speed = speed;
                    }

                    self.queued_ticks = 0;

                    let version = self.version;

                    return Task::perform(task, move |message| Message::Grid(message, version));
                }
            }
            Message::TogglePlayback => {
                self.is_playing = !self.is_playing;
            }
            Message::ToggleGrid(show_grid_lines) => {
                self.grid.toggle_lines(show_grid_lines);
            }
            Message::Clear => {
                self.grid.clear();
                self.version += 1;
            }
            Message::SpeedChanged(speed) => {
                if self.is_playing {
                    self.next_speed = Some(speed.round() as usize);
                } else {
                    self.speed = speed.round() as usize;
                }
            }
            Message::PresetPicked(new_preset) => {
                self.grid = Grid::from_preset(new_preset);
                self.version += 1;
            }
        }

        Task::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        if self.is_playing {
            time::every(Duration::from_millis(1000 / self.speed as u64)).map(|_| Message::Tick)
        } else {
            Subscription::none()
        }
    }

    fn view(&self) -> Element<Message> {
        let version = self.version;
        let selected_speed = self.next_speed.unwrap_or(self.speed);
        let controls = view_controls(
            self.is_playing,
            self.grid.are_lines_visible(),
            selected_speed,
            self.grid.preset(),
        );

        let content = column![
            self.grid
                .view()
                .map(move |message| Message::Grid(message, version)),
            controls,
        ]
        .height(Fill);

        container(content).width(Fill).height(Fill).into()
    }
}

impl Default for LifeSim {
    fn default() -> Self {
        Self::new()
    }
}

fn view_controls<'a>(
    is_playing: bool,
    is_grid_enabled: bool,
    speed: usize,
    preset: Preset,
) -> Element<'a, Message> {
    let playback_controls = row![
        button(if is_playing { "Pause" } else { "Play" }).on_press(Message::TogglePlayback),
        button("Next")
            .on_press(Message::Next)
            .style(button::secondary),
    ]
    .spacing(10);

    let speed_controls = row![
        slider(1.0..=1000.0, speed as f32, Message::SpeedChanged),
        text!("x{speed}").size(16),
    ]
    .align_y(Center)
    .spacing(10);

    row![
        playback_controls,
        speed_controls,
        checkbox("Grid", is_grid_enabled).on_toggle(Message::ToggleGrid),
        row![
            pick_list(preset::ALL, Some(preset), Message::PresetPicked),
            button("Clear")
                .on_press(Message::Clear)
                .style(button::danger)
        ]
        .spacing(10)
    ]
    .padding(10)
    .spacing(20)
    .align_y(Center)
    .into()
}

mod grid {
    use crate::Preset;
    use iced::alignment;
    use iced::mouse;
    use iced::touch;
    use iced::widget::canvas;
    use iced::widget::canvas::event::{self, Event};
    use iced::widget::canvas::{Cache, Canvas, Frame, Geometry, Path, Text};
    use iced::{Color, Element, Fill, Point, Rectangle, Renderer, Size, Theme, Vector};
    use rand::Rng;
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::collections::VecDeque;
    use std::future::Future;
    use std::ops::RangeInclusive;
    use std::time::{Duration, Instant};

    pub struct Grid {
        state: State,
        preset: Preset,
        life_cache: Cache,
        grid_cache: Cache,
        translation: Vector,
        scaling: f32,
        show_lines: bool,
        last_tick_duration: Duration,
        last_queued_ticks: usize,
    }

    #[derive(Debug, Clone)]
    pub enum Message {
        Populate(Cell),
        Unpopulate(Cell),
        Translated(Vector),
        Scaled(f32, Option<Vector>),
        Ticked {
            result: Result<Life, TickError>,
            tick_duration: Duration,
        },
    }

    #[derive(Debug, Clone)]
    pub enum TickError {
        JoinFailed,
    }

    impl Default for Grid {
        fn default() -> Self {
            Self::from_preset(Preset::default())
        }
    }

    impl std::hash::Hash for Organism {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            for cell in &self.cells {
                cell.hash(state);
            }
        }
    }

    impl Grid {
        const MIN_SCALING: f32 = 0.1;
        const MAX_SCALING: f32 = 2.0;

        pub fn from_preset(preset: Preset) -> Self {
            let cells: Vec<Cell> = preset
                .life()
                .into_iter()
                .map(|(i, j)| Cell {
                    i,
                    j,
                    cell_type: match Life::chance(90.0) {
                        true => CellType::Alive,
                        false => CellType::Grower,
                    },
                })
                .collect();

            let mut organisms = Life::find_all_organisms(cells.clone());

            Self {
                state: State::with_life(Life {
                    cells,
                    organisms: organisms,
                }),
                preset,
                life_cache: Cache::default(),
                grid_cache: Cache::default(),
                translation: Vector::default(),
                scaling: 1.0,
                show_lines: true,
                last_tick_duration: Duration::default(),
                last_queued_ticks: 0,
            }
        }

        pub fn tick(&mut self, amount: usize) -> Option<impl Future<Output = Message>> {
            let tick = self.state.tick(amount)?;

            self.last_queued_ticks = amount;

            Some(async move {
                let start = Instant::now();
                let result = tick.await;
                let tick_duration = start.elapsed() / amount as u32;

                Message::Ticked {
                    result,
                    tick_duration,
                }
            })
        }

        pub fn update(&mut self, message: Message) {
            match message {
                Message::Populate(cell) => {
                    self.state.populate(cell);
                    self.life_cache.clear();

                    self.preset = Preset::Custom;
                }
                Message::Unpopulate(cell) => {
                    self.state.unpopulate(&cell);
                    self.life_cache.clear();

                    self.preset = Preset::Custom;
                }
                Message::Translated(translation) => {
                    self.translation = translation;

                    self.life_cache.clear();
                    self.grid_cache.clear();
                }
                Message::Scaled(scaling, translation) => {
                    self.scaling = scaling;

                    if let Some(translation) = translation {
                        self.translation = translation;
                    }

                    self.life_cache.clear();
                    self.grid_cache.clear();
                }
                Message::Ticked {
                    result: Ok(life),
                    tick_duration,
                } => {
                    self.state.update(life);
                    self.life_cache.clear();

                    self.last_tick_duration = tick_duration;
                }
                Message::Ticked {
                    result: Err(error), ..
                } => {
                    dbg!(error);
                }
            }
        }

        pub fn view(&self) -> Element<Message> {
            Canvas::new(self).width(Fill).height(Fill).into()
        }

        pub fn clear(&mut self) {
            self.state = State::default();
            self.preset = Preset::Custom;

            self.life_cache.clear();
        }

        pub fn preset(&self) -> Preset {
            self.preset
        }

        pub fn toggle_lines(&mut self, enabled: bool) {
            self.show_lines = enabled;
        }

        pub fn are_lines_visible(&self) -> bool {
            self.show_lines
        }

        fn visible_region(&self, size: Size) -> Region {
            let width = size.width / self.scaling;
            let height = size.height / self.scaling;

            Region {
                x: -self.translation.x - width / 2.0,
                y: -self.translation.y - height / 2.0,
                width,
                height,
            }
        }

        fn project(&self, position: Point, size: Size) -> Point {
            let region = self.visible_region(size);

            Point::new(
                position.x / self.scaling + region.x,
                position.y / self.scaling + region.y,
            )
        }
    }

    impl canvas::Program<Message> for Grid {
        type State = Interaction;

        fn update(
            &self,
            interaction: &mut Interaction,
            event: Event,
            bounds: Rectangle,
            cursor: mouse::Cursor,
        ) -> (event::Status, Option<Message>) {
            if let Event::Mouse(mouse::Event::ButtonReleased(_)) = event {
                *interaction = Interaction::None;
            }

            let Some(cursor_position) = cursor.position_in(bounds) else {
                return (event::Status::Ignored, None);
            };

            let cell = Cell::at(self.project(cursor_position, bounds.size()));
            let is_populated = self.state.contains(&cell);

            let (populate, unpopulate) = if is_populated {
                (None, Some(Message::Unpopulate(cell)))
            } else {
                (Some(Message::Populate(cell)), None)
            };

            match event {
                Event::Touch(touch::Event::FingerMoved { .. }) => {
                    let message = {
                        *interaction = if is_populated {
                            Interaction::Erasing
                        } else {
                            Interaction::Drawing
                        };

                        populate.or(unpopulate)
                    };

                    (event::Status::Captured, message)
                }
                Event::Mouse(mouse_event) => match mouse_event {
                    mouse::Event::ButtonPressed(button) => {
                        let message = match button {
                            mouse::Button::Left => {
                                *interaction = if is_populated {
                                    Interaction::Erasing
                                } else {
                                    Interaction::Drawing
                                };

                                populate.or(unpopulate)
                            }
                            mouse::Button::Right => {
                                *interaction = Interaction::Panning {
                                    translation: self.translation,
                                    start: cursor_position,
                                };

                                None
                            }
                            _ => None,
                        };

                        (event::Status::Captured, message)
                    }
                    mouse::Event::CursorMoved { .. } => {
                        let message = match *interaction {
                            Interaction::Drawing => populate,
                            Interaction::Erasing => unpopulate,
                            Interaction::Panning { translation, start } => {
                                Some(Message::Translated(
                                    translation + (cursor_position - start) * (1.0 / self.scaling),
                                ))
                            }
                            Interaction::None => None,
                        };

                        let event_status = match interaction {
                            Interaction::None => event::Status::Ignored,
                            _ => event::Status::Captured,
                        };

                        (event_status, message)
                    }
                    mouse::Event::WheelScrolled { delta } => match delta {
                        mouse::ScrollDelta::Lines { y, .. }
                        | mouse::ScrollDelta::Pixels { y, .. } => {
                            if y < 0.0 && self.scaling > Self::MIN_SCALING
                                || y > 0.0 && self.scaling < Self::MAX_SCALING
                            {
                                let old_scaling = self.scaling;

                                let scaling = (self.scaling * (1.0 + y / 30.0))
                                    .clamp(Self::MIN_SCALING, Self::MAX_SCALING);

                                let translation = if let Some(cursor_to_center) =
                                    cursor.position_from(bounds.center())
                                {
                                    let factor = scaling - old_scaling;

                                    Some(
                                        self.translation
                                            - Vector::new(
                                                cursor_to_center.x * factor
                                                    / (old_scaling * old_scaling),
                                                cursor_to_center.y * factor
                                                    / (old_scaling * old_scaling),
                                            ),
                                    )
                                } else {
                                    None
                                };

                                (
                                    event::Status::Captured,
                                    Some(Message::Scaled(scaling, translation)),
                                )
                            } else {
                                (event::Status::Captured, None)
                            }
                        }
                    },
                    _ => (event::Status::Ignored, None),
                },
                _ => (event::Status::Ignored, None),
            }
        }

        fn draw(
            &self,
            _interaction: &Interaction,
            renderer: &Renderer,
            _theme: &Theme,
            bounds: Rectangle,
            cursor: mouse::Cursor,
        ) -> Vec<Geometry> {
            let center = Vector::new(bounds.width / 2.0, bounds.height / 2.0);

            let life = self.life_cache.draw(renderer, bounds.size(), |frame| {
                let background = Path::rectangle(Point::ORIGIN, frame.size());
                frame.fill(&background, Color::from_rgb8(0x40, 0x44, 0x4B));

                frame.with_save(|frame| {
                    frame.translate(center);
                    frame.scale(self.scaling);
                    frame.translate(self.translation);
                    frame.scale(Cell::SIZE);

                    let region = self.visible_region(frame.size());

                    for cell in region.cull(self.state.cells()) {
                        if cell.cell_type == CellType::Alive {
                            frame.fill_rectangle(
                                Point::new(cell.j as f32, cell.i as f32),
                                Size::UNIT,
                                Color::WHITE,
                            );
                        } else if cell.cell_type == CellType::Food {
                            frame.fill_rectangle(
                                Point::new(cell.j as f32, cell.i as f32),
                                Size::UNIT,
                                Color::BLACK,
                            );
                        } else if cell.cell_type == CellType::Grower {
                            frame.fill_rectangle(
                                Point::new(cell.j as f32, cell.i as f32),
                                Size::UNIT,
                                Color::from_rgb(50.0, 27.0, 0.0),
                            )
                        }
                    }
                });
            });

            let overlay = {
                let mut frame = Frame::new(renderer, bounds.size());

                let hovered_cell = cursor
                    .position_in(bounds)
                    .map(|position| Cell::at(self.project(position, frame.size())));

                if let Some(cell) = hovered_cell {
                    frame.with_save(|frame| {
                        frame.translate(center);
                        frame.scale(self.scaling);
                        frame.translate(self.translation);
                        frame.scale(Cell::SIZE);

                        frame.fill_rectangle(
                            Point::new(cell.j as f32, cell.i as f32),
                            Size::UNIT,
                            Color {
                                a: 0.5,
                                ..Color::BLACK
                            },
                        );
                    });
                }

                let text = Text {
                    color: Color::WHITE,
                    size: 14.0.into(),
                    position: Point::new(frame.width(), frame.height()),
                    horizontal_alignment: alignment::Horizontal::Right,
                    vertical_alignment: alignment::Vertical::Bottom,
                    ..Text::default()
                };

                if let Some(cell) = hovered_cell {
                    frame.fill_text(Text {
                        content: format!("({}, {})", cell.j, cell.i),
                        position: text.position - Vector::new(0.0, 16.0),
                        ..text
                    });
                }

                let cell_count = self.state.cell_count();
                let organism_count = self.state.life.organisms.len();

                frame.fill_text(Text {
                    content: format!(
                        "{organism_count} organisms, {cell_count} cell{} @ {:?} ({})",
                        if cell_count == 1 { "" } else { "s" },
                        self.last_tick_duration,
                        self.last_queued_ticks
                    ),
                    ..text
                });
                /*
                frame.fill_text(Text {
                    content: format!("{organism_count}"),
                    ..text
                });
                */

                frame.into_geometry()
            };

            if self.scaling >= 0.2 && self.show_lines {
                let grid = self.grid_cache.draw(renderer, bounds.size(), |frame| {
                    frame.translate(center);
                    frame.scale(self.scaling);
                    frame.translate(self.translation);
                    frame.scale(Cell::SIZE);

                    let region = self.visible_region(frame.size());
                    let rows = region.rows();
                    let columns = region.columns();
                    let (total_rows, total_columns) =
                        (rows.clone().count(), columns.clone().count());
                    let width = 2.0 / Cell::SIZE as f32;
                    let color = Color::from_rgb8(70, 74, 83);

                    frame.translate(Vector::new(-width / 2.0, -width / 2.0));

                    for row in region.rows() {
                        frame.fill_rectangle(
                            Point::new(*columns.start() as f32, row as f32),
                            Size::new(total_columns as f32, width),
                            color,
                        );
                    }

                    for column in region.columns() {
                        frame.fill_rectangle(
                            Point::new(column as f32, *rows.start() as f32),
                            Size::new(width, total_rows as f32),
                            color,
                        );
                    }
                });

                vec![life, grid, overlay]
            } else {
                vec![life, overlay]
            }
        }

        fn mouse_interaction(
            &self,
            interaction: &Interaction,
            bounds: Rectangle,
            cursor: mouse::Cursor,
        ) -> mouse::Interaction {
            match interaction {
                Interaction::Drawing => mouse::Interaction::Crosshair,
                Interaction::Erasing => mouse::Interaction::Crosshair,
                Interaction::Panning { .. } => mouse::Interaction::Grabbing,
                Interaction::None if cursor.is_over(bounds) => mouse::Interaction::Crosshair,
                Interaction::None => mouse::Interaction::default(),
            }
        }
    }

    #[derive(Default)]
    struct State {
        life: Life,
        births: FxHashSet<Cell>,
        is_ticking: bool,
    }

    impl State {
        pub fn with_life(life: Life) -> Self {
            Self {
                life,
                ..Self::default()
            }
        }

        fn cell_count(&self) -> usize {
            self.life.len() + self.births.len()
        }

        fn contains(&self, cell: &Cell) -> bool {
            self.life.contains(cell) || self.births.contains(cell)
        }

        fn cells(&self) -> impl Iterator<Item = &Cell> {
            self.life.iter().chain(self.births.iter())
        }

        fn populate(&mut self, cell: Cell) {
            if self.is_ticking {
                self.births.insert(cell);
            } else {
                self.life.populate(cell);
            }
        }

        fn unpopulate(&mut self, cell: &Cell) {
            if self.is_ticking {
                let _ = self.births.remove(cell);
            } else {
                self.life.unpopulate(cell);
            }
        }

        fn update(&mut self, mut life: Life) {
            self.births.drain().for_each(|cell| life.populate(cell));

            self.life = life;
            self.is_ticking = false;
        }

        fn tick(&mut self, amount: usize) -> Option<impl Future<Output = Result<Life, TickError>>> {
            if self.is_ticking {
                return None;
            }

            self.is_ticking = true;

            let mut life = self.life.clone();

            Some(async move {
                tokio::task::spawn_blocking(move || {
                    for _ in 0..amount {
                        life.tick();
                    }

                    life
                })
                .await
                .map_err(|_| TickError::JoinFailed)
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Organism {
        cells: Vec<Cell>,
        energy: usize,
        able_to_move: bool,
    }

    impl Organism {
        const NEW_ORGANISM_ENERGY: usize = 100;

        fn new(cells: Vec<Cell>) -> Self {
            Organism {
                cells: cells.clone(),
                energy: Organism::NEW_ORGANISM_ENERGY,
                able_to_move: true, // Organism::check_if_can_move(cells),
            }
        }

        fn check_if_can_move(cells: Vec<Cell>) -> bool {
            for cell in cells {
                if cell.cell_type == CellType::Mover {
                    return true;
                }
            }
            false
        }

        fn len(&self) -> usize {
            self.cells.len()
        }
        pub fn contains(&self, cell: &Cell) -> bool {
            self.cells.contains(cell)
        }

        fn populate(&mut self, cell: Cell) {
            if !self.contains(&cell) {
                self.cells.push(cell)
            }
        }

        fn unpopulate(&mut self, cell: &Cell) {
            self.cells.retain(|c| c != cell);
        }

        // want to create a way for food to be eaten, organism to reproduce and randomly evolve.

        pub fn adjacent_food(&self, life: &Life) -> Vec<Cell> {
            let mut food = Vec::new();
            for cell in &self.cells {
                for neighbor in Cell::neighbors(*cell) {
                    if let Some(food_cell) = life.cells.iter().find(|c| **c == neighbor) {
                        if food_cell.cell_type == CellType::Food && !self.contains(food_cell) {
                            if !food.contains(food_cell) {
                                food.push(*food_cell);
                            }
                        }
                    }
                }
            }
            food
        }

        /// Helper: Compute the bounding box of the organism as (min_j, max_j, min_i, max_i)
        pub fn bounding_box(&self) -> (isize, isize, isize, isize) {
            let min_j = self.cells.iter().map(|c| c.j).min().unwrap_or(0);
            let max_j = self.cells.iter().map(|c| c.j).max().unwrap_or(0);
            let min_i = self.cells.iter().map(|c| c.i).min().unwrap_or(0);
            let max_i = self.cells.iter().map(|c| c.i).max().unwrap_or(0);
            (min_j, max_j, min_i, max_i)
        }
    }

    // Need to find a way to get the Hash set of all possible cells, as a concatenation of organisms' cells and empty cells.

    #[derive(Clone, Default)]
    pub struct Life {
        cells: Vec<Cell>,
        organisms: Vec<Organism>,
    }

    impl Life {
        const CHANCE_TO_EAT: f64 = 80.0;
        const MUTATION_RATE: f64 = 50.0;
        const CHANCE_TO_SKIP_REPRODUCTION: f64 = 60.0;
        const CHANCE_TO_GROW_FOOD: f64 = 1.0;
        const ENERGY_FROM_FOOD: usize = 5;

        fn len(&self) -> usize {
            self.cells.len()
        }

        fn contains(&self, cell: &Cell) -> bool {
            self.cells.contains(cell)
        }

        fn populate(&mut self, cell: Cell) {
            if !self.contains(&cell) {
                self.cells.push(cell);
            }
        }

        fn unpopulate(&mut self, cell: &Cell) {
            self.cells.retain(|c| c != cell);
        }

        fn create_organism_from_cell_clump(cells: Vec<Cell>, start_cell: Cell) -> Organism {
            let mut organism_cells: Vec<Cell> = Vec::new();

            let mut visited = Vec::new();
            let mut queue = VecDeque::new();

            queue.push_back(start_cell);
            visited.push(start_cell);

            while let Some(cell) = queue.pop_front() {
                organism_cells.push(cell);

                // Define possible movement directions (up, down, left, right)
                let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

                for &(dx, dy) in &directions {
                    let neighbor = Cell {
                        i: cell.i + dy,
                        j: cell.j + dx,
                        cell_type: cell.cell_type, // Assume same type for simplicity
                    };

                    // Check if neighbor is a valid cell & not visited
                    if cells.contains(&neighbor) && !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                        visited.push(neighbor);
                    }
                }
            }
            Organism::new(organism_cells)
        }

        pub fn grow_food(&mut self) {
            let mut new_food_cells = Vec::new();

            for cell in &self.cells {
                if cell.cell_type == CellType::Grower {
                    let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

                    for &(dx, dy) in &directions {
                        let new_cell = Cell {
                            i: cell.i + dy,
                            j: cell.j + dx,
                            cell_type: CellType::Food,
                        };

                        // Check if this position is empty
                        if !self
                            .cells
                            .iter()
                            .any(|c| c.i == new_cell.i && c.j == new_cell.j)
                        {
                            if Life::chance(Life::CHANCE_TO_GROW_FOOD) {
                                return;
                            }
                            new_food_cells.push(new_cell);
                            break; // Grow only one food per cycle
                        }
                    }
                }
            }

            // Add new food cells to the world
            self.cells.extend(new_food_cells);
        }

        fn chance(percentage: f64) -> bool {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            rng.gen::<f64>() < (percentage / 100.0)
        }

        fn consume_food(&mut self) {
            let mut new_cells = self.cells.clone();

            for organism in &mut self.organisms {
                let mut food_cells_to_remove = Vec::new();

                for cell in &organism.cells {
                    let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

                    for &(dx, dy) in &directions {
                        let neighbor = Cell {
                            i: cell.i + dy,
                            j: cell.j + dx,
                            cell_type: CellType::Food,
                        };

                        // If a food cell is found in the world
                        if let Some(pos) = new_cells.iter().position(|c| *c == neighbor) {
                            if Life::chance(Life::CHANCE_TO_EAT) {
                                food_cells_to_remove.push(pos);
                                organism.energy += Self::ENERGY_FROM_FOOD; // Increase energy
                            }
                        }
                    }
                }

                // Remove food cells from the world
                for &pos in food_cells_to_remove.iter().rev() {
                    new_cells.remove(pos);
                }
            }

            self.cells = new_cells;
        }

        fn find_all_organisms(cells: Vec<Cell>) -> Vec<Organism> {
            let mut organisms = Vec::new();
            let mut visited = Vec::new();

            for &cell in &cells {
                if visited.contains(&cell) {
                    continue; // Skip already-processed cells
                }

                let mut organism_cells = Vec::new();
                let mut queue = VecDeque::new();

                queue.push_back(cell);
                visited.push(cell);

                while let Some(current) = queue.pop_front() {
                    organism_cells.push(current);

                    let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

                    for &(dx, dy) in &directions {
                        let neighbor = Cell {
                            i: current.i + dy,
                            j: current.j + dx,
                            cell_type: current.cell_type,
                        };

                        // Check if it's a valid cell and hasn't been visited
                        if cells.contains(&neighbor) && !visited.contains(&neighbor) {
                            queue.push_back(neighbor);
                            visited.push(neighbor);
                        }
                    }
                }

                organisms.push(Organism::new(organism_cells));
            }

            organisms
        }

        pub fn move_organisms(&mut self) {
            use rand::Rng;
            let mut new_organisms = Vec::new();
            let mut new_cells = self.cells.clone(); // Keep all existing cells

            let mut rng = rand::thread_rng();

            for organism in &self.organisms {
                /*
                if !organism.able_to_move {
                    continue;
                }
                */
                let mut new_organism_cells = Vec::new();
                let mut can_move = true;
                let mut energy_change: usize = 0;

                // Random movement: right (+1, 0), left (-1, 0), up (0, -1), down (0, +1)
                let (dx, dy) = match rng.gen_range(0..4) {
                    0 => (1, 0),  // Move right
                    1 => (-1, 0), // Move left
                    2 => (0, 1),  // Move down
                    _ => (0, -1), // Move up
                };

                // Check if all cells can move in the chosen direction
                for cell in &organism.cells {
                    let potential_new_position = Cell {
                        i: cell.i + dy,
                        j: cell.j + dx,
                        cell_type: cell.cell_type,
                    };

                    // Ensure the destination is not occupied by another organism or food (unless it's part of itself)
                    if self.cells.iter().any(|existing_cell| {
                        existing_cell.i == potential_new_position.i
                            && existing_cell.j == potential_new_position.j
                            && existing_cell.cell_type != CellType::Food
                            && !organism.contains(existing_cell)
                    }) {
                        can_move = false;
                        break;
                    }
                }

                if can_move {
                    // Remove old organism cells from `new_cells`
                    for cell in &organism.cells {
                        new_cells.retain(|c| c != cell);
                    }

                    // Move the organism cells to new positions
                    for cell in &organism.cells {
                        let new_cell = Cell {
                            i: cell.i + dy,
                            j: cell.j + dx,
                            cell_type: cell.cell_type,
                        };
                        new_organism_cells.push(new_cell);
                        new_cells.push(new_cell);
                    }
                    energy_change = 1
                } else {
                    // If movement is not possible, keep the organism in place
                    for cell in &organism.cells {
                        new_organism_cells.push(*cell);
                        if !new_cells.contains(cell) {
                            new_cells.push(*cell);
                        }
                    }
                    energy_change = 0;
                }
                new_organisms.push(Organism {
                    cells: new_organism_cells,
                    energy: organism.energy - 1,
                    able_to_move: organism.able_to_move,
                });
            }

            // Update the world's state
            self.organisms = new_organisms;
            self.cells = new_cells;
        }
        fn tick(&mut self) {
            // Phase 1: Move organisms

            self.grow_food();
            self.move_organisms();
            self.consume_food();
            self.cull_dead_organisms();
            self.reproduce_organism_test_again();
        }

        pub fn iter(&self) -> impl Iterator<Item = &Cell> {
            self.cells.iter()
        }

        /// Reproduction cost: energy that must be expended to reproduce.
        const REPRODUCTION_COST: usize = 200;
        /// The gap (in cell units) between the parent and the offspring.
        const REPRODUCTION_GAP: isize = 1;

        /// For each organism that has energy ≥ 3, try to reproduce by placing an offspring
        /// next to it (left, right, up, or down) provided the destination area is empty.
        pub fn reproduce_organisms(&mut self) {
            // Collect new organisms in a temporary vector.
            let mut new_organisms = Vec::new();

            for organism in &mut self.organisms {
                if organism.energy < Self::REPRODUCTION_COST {
                    continue;
                }

                if Life::chance(Self::CHANCE_TO_SKIP_REPRODUCTION) {
                    continue;
                }

                // Compute the parent's bounding box
                let (min_j, max_j, min_i, max_i) = organism.bounding_box();
                let width = max_j - min_j + 1;
                let height = max_i - min_i + 1;

                // Define the candidate offsets for reproduction:
                // right, left, down, up. The offset is parent's dimension plus a gap.
                let candidates = [
                    (width + Self::REPRODUCTION_GAP, 0),     // right
                    (-(width + Self::REPRODUCTION_GAP), 0),  // left
                    (0, height + Self::REPRODUCTION_GAP),    // down
                    (0, -(height + Self::REPRODUCTION_GAP)), // up
                ];

                // Try each candidate direction until one works.
                let mut reproduced = false;
                for &(dx, dy) in &candidates {
                    // Generate the offspring cells by offsetting each parent's cell.
                    let offspring_cells: Vec<Cell> = organism
                        .cells
                        .iter()
                        .map(|cell| Cell {
                            i: cell.i + dy,
                            j: cell.j + dx,
                            cell_type: cell.cell_type, // Presume the same type (Alive)
                        })
                        .collect();

                    // Check if any of these new cells already exist in the world's grid.
                    // (You might wish to add additional criteria here.)
                    let conflict = offspring_cells
                        .iter()
                        .any(|offspring_cell| self.cells.contains(offspring_cell));
                    if conflict {
                        continue; // try next candidate direction
                    }

                    // No conflict found: create the offspring organism.
                    let offspring = Organism::new(offspring_cells.clone());
                    // Deduct reproduction cost from the parent.
                    organism.energy -= Self::REPRODUCTION_COST;
                    // Add offspring cells to the world's grid.
                    for cell in offspring_cells {
                        self.cells.push(cell);
                    }
                    // Save the new organism.
                    new_organisms.push(offspring);
                    reproduced = true;
                    break;
                }
                // Optionally, if reproduction fails in all directions, you could log or handle that case.
                if !reproduced {
                    // For example: println!("Organism at {:?} could not reproduce due to space constraints.", organism.bounding_box());
                }
            }

            // Add all new organisms to the Life structure.
            self.organisms.extend(new_organisms);
        }

        fn random_mutation() -> Option<CellType> {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let NewType = match rng.gen_range(0..1) {
                0 => CellType::Alive,
                1 => CellType::Grower,
                _ => CellType::Alive,
            };
            Some(NewType)
        }

        pub fn reproduce_organism_test_again(&mut self) {
            // Collect new organisms in a temporary vector.
            let mut new_organisms = Vec::new();

            for organism in &mut self.organisms {
                if organism.energy < Self::REPRODUCTION_COST {
                    continue;
                }

                if Life::chance(70.0) {
                    continue;
                }

                // Compute the parent's bounding box.
                let (min_j, max_j, min_i, max_i) = organism.bounding_box();
                let width = max_j - min_j + 1;
                let height = max_i - min_i + 1;

                // Define candidate offsets for reproduction:
                // right, left, down, up. The offset is parent's dimension plus a gap.
                let candidates = [
                    (width + Self::REPRODUCTION_GAP, 0),     // right
                    (-(width + Self::REPRODUCTION_GAP), 0),  // left
                    (0, height + Self::REPRODUCTION_GAP),    // down
                    (0, -(height + Self::REPRODUCTION_GAP)), // up
                ];

                // Try each candidate direction until one works.
                let mut reproduced = false;
                for &(dx, dy) in &candidates {
                    // Generate the offspring cells by offsetting each parent's cell.
                    let mut offspring_cells: Vec<Cell> = organism
                        .cells
                        .iter()
                        .map(|cell| {
                            let mut new_cell = Cell {
                                i: cell.i + dy,
                                j: cell.j + dx,
                                cell_type: cell.cell_type, // Default to the same type.
                            };

                            // Apply mutation with a set probability.
                            if Life::chance(Self::MUTATION_RATE * 100.0) {
                                if let Some(mutated_type) = Cell::random_type() {
                                    new_cell.cell_type = mutated_type;
                                }
                            }
                            new_cell
                        })
                        .collect();

                    // Check if any of these new cells already exist in the world's grid.
                    let conflict = offspring_cells
                        .iter()
                        .any(|offspring_cell| self.cells.contains(offspring_cell));
                    if conflict {
                        continue; // Try next candidate direction.
                    }

                    // Introduce a random new cell adjacent to the offspring (max once per evolution).
                    if Life::chance(Self::MUTATION_RATE) {
                        if !offspring_cells.is_empty() {
                            // Pick a random cell from the offspring.
                            let random_index =
                                rand::thread_rng().gen_range(0..offspring_cells.len());
                            let base_cell = offspring_cells[random_index];
                            // Allow mutation offsets in all four directions.
                            let mutation_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                            let offset_index =
                                rand::thread_rng().gen_range(0..mutation_offsets.len());
                            let (mx, my) = mutation_offsets[offset_index];
                            let new_cell = Cell {
                                i: base_cell.i + my,
                                j: base_cell.j + mx,
                                cell_type: Cell::random_type().unwrap_or(base_cell.cell_type),
                            };
                            // Only add if that spot is empty.
                            if !self.cells.contains(&new_cell) {
                                offspring_cells.push(new_cell);
                            }
                        }
                    }

                    // No conflict found: create the offspring organism.
                    let offspring = Organism::new(offspring_cells.clone());
                    // Deduct reproduction cost from the parent.
                    organism.energy -= Self::REPRODUCTION_COST;
                    // Add offspring cells to the world's grid.
                    for cell in offspring_cells {
                        self.cells.push(cell);
                    }
                    // Save the new organism.
                    new_organisms.push(offspring);
                    reproduced = true;
                    break;
                }
                // Optionally, if reproduction fails in all directions, you could log or handle that case.
                if !reproduced {
                    // e.g., println!("Organism at {:?} could not reproduce due to space constraints.", organism.bounding_box());
                }
            }

            // Add all new organisms to the Life structure.
            self.organisms.extend(new_organisms);
        }

        pub fn cull_dead_organisms(&mut self) {
            let mut new_food_cells = Vec::new();

            // Iterate through organisms and convert dead ones into food
            self.organisms.retain(|organism| {
                if organism.energy <= 0 {
                    // Convert all its cells into food and add to new food list
                    for cell in &organism.cells {
                        new_food_cells.push(Cell {
                            i: cell.i,
                            j: cell.j,
                            cell_type: CellType::Food,
                        });
                    }
                    false // Remove the organism from self.organisms
                } else {
                    true // Keep the organism if it's still alive
                }
            });

            self.cells.retain(|cell| {
                !new_food_cells
                    .iter()
                    .any(|food| food.i == cell.i && food.j == cell.j)
            });

            self.cells.extend(new_food_cells);
        }
    }

    // I am going to need to impelement a structure for an organism, and instead of cells were gonna have to make separate function to separate organisms
    impl std::iter::FromIterator<Cell> for Life {
        fn from_iter<I: IntoIterator<Item = Cell>>(iter: I) -> Self {
            let organisms: Vec<Organism> = Vec::new();
            Life {
                cells: iter.into_iter().collect(),
                organisms,
            }
        }
    }

    impl std::fmt::Debug for Life {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Life")
                .field("cells", &self.cells.len())
                .finish()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum CellType {
        Empty,
        Alive,
        Food,
        Grower,
        Mover,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Cell {
        i: isize,
        j: isize,
        cell_type: CellType,
    }

    impl Cell {
        const SIZE: u16 = 20;

        fn at(position: Point) -> Cell {
            let i = (position.y / Cell::SIZE as f32).ceil() as isize;
            let j = (position.x / Cell::SIZE as f32).ceil() as isize;

            Cell {
                i: i.saturating_sub(1),
                j: j.saturating_sub(1),
                cell_type: CellType::Food, // we want only to be able to draw food onto the world
            }
        }

        fn cluster(cell: Cell) -> impl Iterator<Item = Cell> {
            use itertools::Itertools;

            let rows = cell.i.saturating_sub(1)..=cell.i.saturating_add(1);
            let columns = cell.j.saturating_sub(1)..=cell.j.saturating_add(1);
            rows.cartesian_product(columns).map(move |(i, j)| Cell {
                i,
                j,
                cell_type: cell.cell_type,
            })
        }

        fn neighbors(cell: Cell) -> impl Iterator<Item = Cell> {
            Cell::cluster(cell).filter(move |candidate| *candidate != cell)
        }

        pub fn is_food(&self) -> bool {
            self.cell_type == CellType::Food
        }

        pub fn random_type() -> Option<CellType> {
            use CellType::*;
            let types = [Alive, Grower, Mover]; // Define available types
            if types.is_empty() {
                None
            } else {
                let random_index = rand::thread_rng().gen_range(0..types.len());
                Some(types[random_index])
            }
        }
    }

    pub struct Region {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    }

    impl Region {
        fn rows(&self) -> RangeInclusive<isize> {
            let first_row = (self.y / Cell::SIZE as f32).floor() as isize;

            let visible_rows = (self.height / Cell::SIZE as f32).ceil() as isize;

            first_row..=first_row + visible_rows
        }

        fn columns(&self) -> RangeInclusive<isize> {
            let first_column = (self.x / Cell::SIZE as f32).floor() as isize;

            let visible_columns = (self.width / Cell::SIZE as f32).ceil() as isize;

            first_column..=first_column + visible_columns
        }

        fn cull<'a>(
            &self,
            cells: impl Iterator<Item = &'a Cell>,
        ) -> impl Iterator<Item = &'a Cell> {
            let rows = self.rows();
            let columns = self.columns();

            cells.filter(move |cell| rows.contains(&cell.i) && columns.contains(&cell.j))
        }
    }

    pub enum Interaction {
        None,
        Drawing,
        Erasing,
        Panning { translation: Vector, start: Point },
    }

    impl Default for Interaction {
        fn default() -> Self {
            Self::None
        }
    }
}
