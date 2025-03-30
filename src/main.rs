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
    use rustc_hash::{FxHashMap, FxHashSet};
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
            let cells: FxHashSet<Cell> = preset
                .life()
                .into_iter()
                .map(|(i, j)| Cell {
                    i,
                    j,
                    cell_type: CellType::Alive,
                })
                .collect();

            let mut organisms = FxHashSet::default();
            organisms.insert(Organism {
                cells: cells.clone(),
                energy: 3,
            });

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

                frame.fill_text(Text {
                    content: format!(
                        "{cell_count} cell{} @ {:?} ({})",
                        if cell_count == 1 { "" } else { "s" },
                        self.last_tick_duration,
                        self.last_queued_ticks
                    ),
                    ..text
                });

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
        cells: FxHashSet<Cell>,
        energy: usize,
    }

    impl Organism {
        fn new(cells: FxHashSet<Cell>) -> Self {
            Organism { cells, energy: 3 }
        }

        fn len(&self) -> usize {
            self.cells.len()
        }
        fn contains(&self, cell: &Cell) -> bool {
            self.cells.contains(cell)
        }

        fn populate(&mut self, cell: Cell) {
            self.cells.insert(cell);
        }

        fn unpopulate(&mut self, cell: &Cell) {
            let _ = self.cells.remove(cell);
        }

        // want to create a way for food to be eaten, organism to reproduce and randomly evolve.
        fn adjacent_food(&self, life: &Life) -> FxHashSet<Cell> {
            let mut food = FxHashSet::default();

            for cell in &self.cells {
                for neighbor in Cell::neighbors(*cell) {
                    if let Some(food_cell) = life.cells.get(&neighbor) {
                        if food_cell.cell_type == CellType::Food {
                            food.insert(*food_cell);
                        }
                    }
                }
            }

            food
        }

        pub fn consume_food(&mut self, life: &mut Life) {
            let mut gained_energy = 0;
            let mut food_to_remove = FxHashSet::default();

            for cell in &self.cells {
                for neighbor in Cell::neighbors(*cell) {
                    if let Some(food_cell) = life.cells.get(&neighbor) {
                        if food_cell.is_food() {
                            gained_energy += 1;
                            food_to_remove.insert(*food_cell);
                        }
                    }
                }
            }

            self.energy += gained_energy;
            for food in food_to_remove {
                life.cells.remove(&food);
            }
        }
    }

    // Need to find a way to get the Hash set of all possible cells, as a concatenation of organisms' cells and empty cells.

    #[derive(Clone, Default)]
    pub struct Life {
        cells: FxHashSet<Cell>,
        organisms: FxHashSet<Organism>,
    }

    impl Life {
        fn len(&self) -> usize {
            self.cells.len()
        }

        fn contains(&self, cell: &Cell) -> bool {
            self.cells.contains(cell)
        }

        fn populate(&mut self, cell: Cell) {
            self.cells.insert(cell);
        }

        fn unpopulate(&mut self, cell: &Cell) {
            let _ = self.cells.remove(cell);
        }

        fn move_organisms(&mut self) {
            use rand::Rng;
            let mut new_organisms = FxHashSet::default();
            let mut new_cells = self.cells.clone(); // Keep all existing cells

            let mut rng = rand::thread_rng();

            for organism in &self.organisms {
                let mut new_organism_cells = FxHashSet::default();
                let mut can_move = true;

                // Random movement: right (+1, 0), left (-1, 0), up (0, -1), down (0, +1)
                let (dx, dy) = match rng.gen_range(0..4) {
                    0 => (1, 0),  // Move right
                    1 => (-1, 0), // Move left
                    2 => (0, 1),  // Move down
                    _ => (0, -1), // Move up
                };

                // First check if organism can move in chosen direction
                for cell in &organism.cells {
                    let potential_new_position = Cell {
                        i: (cell.i as isize + dy),
                        j: (cell.j as isize + dx),
                        cell_type: cell.cell_type,
                    };

                    // Check if destination is already occupied by a cell that doesn't belong to this organism
                    for existing_cell in &self.cells {
                        if existing_cell.i == potential_new_position.i
                            && existing_cell.j == potential_new_position.j
                            && existing_cell.cell_type == CellType::Food
                            && !organism.contains(existing_cell)
                        {
                            // Destination contains food and is not part of this organism
                            can_move = false;
                            break;
                        }
                    }

                    if !can_move {
                        break;
                    }
                }

                if can_move {
                    // Remove old organism cells
                    for cell in &organism.cells {
                        new_cells.remove(cell);
                    }

                    // Add cells at new positions
                    for cell in &organism.cells {
                        let new_cell = Cell {
                            i: (cell.i as isize + dy),
                            j: (cell.j as isize + dx),
                            cell_type: cell.cell_type,
                        };

                        new_organism_cells.insert(new_cell);
                        new_cells.insert(new_cell);
                    }
                } else {
                    // Cannot move, keep organism in place
                    for cell in &organism.cells {
                        new_organism_cells.insert(*cell);
                        // Make sure to keep the existing cells in the new_cells collection too
                        new_cells.insert(*cell);
                    }
                }

                new_organisms.insert(Organism::new(new_organism_cells));
            }

            self.organisms = new_organisms;
            self.cells = new_cells;
        }

        /*

        const REPRODUCTION_ENERGY: usize = 3;

        fn handle_reproduction(&mut self) {
            let mut new_organisms = FxHashSet::default();
            let mut cells_to_remove = FxHashSet::default();

            for organism in &self.organisms {
                if organism.energy < Self::REPRODUCTION_ENERGY {
                    new_organisms.insert(organism.clone());
                    continue;
                }

                if let Some((offspring, consumed_food)) = self.try_reproduce(organism) {
                    // Add offspring and update parent
                    new_organisms.insert(offspring);
                    let mut parent = organism.clone();
                    parent.energy -= Self::REPRODUCTION_ENERGY;
                    new_organisms.insert(parent);

                    // Mark food for removal
                    cells_to_remove.extend(consumed_food);
                } else {
                    new_organisms.insert(organism.clone());
                }
            }

            // Update game state
            self.organisms = new_organisms;
            self.cells.retain(|cell| !cells_to_remove.contains(cell));
        }

        fn try_reproduce(&self, parent: &Organism) -> Option<(Organism, FxHashSet<Cell>)> {
            use rand::prelude::SliceRandom;
            let (width, height) = self.organism_dimensions(parent);
            let directions = [
                (1, 0, width),   // Right (needs width spacing)
                (-1, 0, width),  // Left (needs width spacing)
                (0, 1, height),  // Down (needs height spacing)
                (0, -1, height), // Up (needs height spacing)
            ];

            let mut rng = rand::thread_rng();
            let mut directions = directions.to_vec();
            directions.shuffle(&mut rng);

            for (dx, dy, spacing) in directions {
                if let Some((offspring, food)) =
                    self.check_reproduction_space(parent, dx, dy, spacing)
                {
                    return Some((offspring, food));
                }
            }

            None
        }

        fn organism_dimensions(&self, organism: &Organism) -> (isize, isize) {
            let (min_j, max_j, min_i, max_i) = organism.cells.iter().fold(
                (isize::MAX, isize::MIN, isize::MAX, isize::MIN),
                |(min_j, max_j, min_i, max_i), cell| {
                    (
                        min_j.min(cell.j),
                        max_j.max(cell.j),
                        min_i.min(cell.i),
                        max_i.max(cell.i),
                    )
                },
            );

            let width = (max_j - min_j + 1).max(1);
            let height = (max_i - min_i + 1).max(1);

            (width, height)
        }

        fn check_reproduction_space(
            &self,
            parent: &Organism,
            dx: isize,
            dy: isize,
            spacing: isize,
        ) -> Option<(Organism, FxHashSet<Cell>)> {
            let mut offspring_cells = FxHashSet::default();
            let mut consumed_food = FxHashSet::default();

            // Check all cells in reproduction path
            for cell in &parent.cells {
                for step in 1..=spacing {
                    let check_cell = Cell {
                        j: cell.j + dx * step,
                        i: cell.i + dy * step,
                        cell_type: CellType::Alive,
                    };

                    if let Some(existing) = self.cells.get(&check_cell) {
                        match existing.cell_type {
                            CellType::Alive => return None, // Collision with living cell
                            CellType::Food => {
                                consumed_food.insert(check_cell);
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Create offspring cells if path is clear
            for cell in &parent.cells {
                let new_cell = Cell {
                    j: cell.j + dx * spacing,
                    i: cell.i + dy * spacing,
                    cell_type: CellType::Alive,
                };
                offspring_cells.insert(new_cell);
            }

            // Require at least one food cell consumed
            if !consumed_food.is_empty() {
                Some((
                    Organism {
                        cells: offspring_cells,
                        energy: 1,
                    },
                    consumed_food,
                ))
            } else {
                None
            }
        }
        */
        /*

        fn handle_eating_and_reproduction(&mut self) {
            let mut new_organisms = FxHashSet::default();
            let mut food_to_remove = Vec::new();

            // Process each organism for eating food
            for organism in &self.organisms {
                let mut updated_organism = organism.clone();

                // Find food that's adjacent to any organism cell
                for cell in &organism.cells {
                    for neighbor in Cell::neighbors(*cell) {
                        // Check if this neighbor is a food cell in our global cells collection
                        if let Some(food_cell) = self.cells.get(&neighbor) {
                            if food_cell.cell_type == CellType::Food {
                                // Found food, mark for removal and add energy
                                food_to_remove.push(*food_cell);
                                updated_organism.energy += 1;
                                // Log for debugging
                                println!(
                                    "Organism found food at ({}, {}), energy now: {}",
                                    food_cell.i, food_cell.j, updated_organism.energy
                                );
                            }
                        }
                    }
                }

                // Check if organism has enough energy to reproduce
                if updated_organism.energy >= 3 {
                    if let Some(offspring) = self.try_reproduce(&updated_organism) {
                        // Log reproduction
                        println!(
                            "Organism reproduced! New organism with {} cells",
                            offspring.cells.len()
                        );

                        // Add offspring cells to global cells collection
                        for cell in &offspring.cells {
                            self.cells.insert(*cell);
                        }

                        // Add offspring to new organisms list
                        new_organisms.insert(offspring);

                        // Reduce parent's energy
                        updated_organism.energy -= 3;
                    }
                }

                // Add updated organism to new set
                new_organisms.insert(updated_organism);
            }
        }

        // Helper method for the Life struct to handle reproduction

        fn try_reproduce(&self, parent: &Organism) -> Option<Organism> {
            use rand::prelude::SliceRandom;
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Directions to try for reproduction
            let directions = [(1, 0), (-1, 0), (0, 1), (0, -1)];

            // Try each direction randomly
            let mut shuffled_dirs = directions.to_vec();
            shuffled_dirs.shuffle(&mut rng);

            for (dx, dy) in shuffled_dirs {
                let mut offspring_cells = FxHashSet::default();
                let mut can_place = true;

                // For each cell in parent, create corresponding cell in offspring
                for cell in &parent.cells {
                    let new_i = cell.i + dy;
                    let new_j = cell.j + dx;
                    let new_cell = Cell {
                        i: new_i,
                        j: new_j,
                        cell_type: CellType::Alive,
                    };

                    // Check if position is already occupied
                    if self.cells.contains(&new_cell) {
                        can_place = false;
                        break;
                    }

                    offspring_cells.insert(new_cell);
                }

                if can_place && !offspring_cells.is_empty() {
                    return Some(Organism {
                        cells: offspring_cells,
                        energy: 1, // Start with minimal energy
                    });
                }
            }

            None
        }

        // Fixed adjacent_food method in Organism struct
        fn adjacent_food(&self, life: &Life) -> FxHashSet<Cell> {
            let mut food_cells = FxHashSet::default();

            for cell in &self.cells {
                for neighbor in Cell::neighbors(*cell) {
                    // Check if this neighbor is a food cell
                    if life.cells.contains(&neighbor) && neighbor.cell_type == CellType::Food {
                        food_cells.insert(neighbor);
                    }
                }
            }

            food_cells
        }
        */

        const REPRODUCTION_COST: usize = 3;
        const REPRODUCTION_PADDING: isize = 1;

        fn tick(&mut self) {
            // Phase 1: Move organisms
            self.move_organisms();

            // Phase 2: Consume food and gain energy
            let mut organisms: Vec<Organism> = self.organisms.drain().collect();
            for organism in &mut organisms {
                organism.consume_food(self);
            }
            self.organisms.extend(organisms);

            // Phase 3: Handle reproduction
            self.handle_reproduction();
        }

        fn handle_reproduction(&mut self) {
            let mut new_organisms = FxHashSet::default();
            let mut cells_to_remove = FxHashSet::default();

            for organism in &self.organisms {
                if organism.energy < Self::REPRODUCTION_COST {
                    new_organisms.insert(organism.clone());
                    continue;
                }

                if let Some((offspring, consumed_food)) = self.try_reproduce(organism) {
                    new_organisms.insert(offspring);
                    let mut parent = organism.clone();
                    parent.energy -= Self::REPRODUCTION_COST;
                    new_organisms.insert(parent);
                    cells_to_remove.extend(consumed_food);
                } else {
                    new_organisms.insert(organism.clone());
                }
            }

            self.organisms = new_organisms;
            self.cells.retain(|c| !cells_to_remove.contains(c));
        }

        fn try_reproduce(&self, parent: &Organism) -> Option<(Organism, FxHashSet<Cell>)> {
            let (min_j, max_j, min_i, max_i) = self.organism_bounding_box(parent);
            let width = max_j - min_j + 1;
            let height = max_i - min_i + 1;

            let directions = [
                (1, 0, width),   // Right (needs width spacing)
                (-1, 0, width),  // Left (needs width spacing)
                (0, 1, height),  // Down (needs height spacing)
                (0, -1, height), // Up (needs height spacing)
            ];

            for (dx, dy, spacing) in directions {
                if let Some(result) =
                    self.check_reproduction(parent, dx, dy, spacing + Self::REPRODUCTION_PADDING)
                {
                    return Some(result);
                }
            }
            None
        }

        fn organism_bounding_box(&self, organism: &Organism) -> (isize, isize, isize, isize) {
            organism.cells.iter().fold(
                (isize::MAX, isize::MIN, isize::MAX, isize::MIN),
                |(min_j, max_j, min_i, max_i), cell| {
                    (
                        min_j.min(cell.j),
                        max_j.max(cell.j),
                        min_i.min(cell.i),
                        max_i.max(cell.i),
                    )
                },
            )
        }

        fn check_reproduction(
            &self,
            parent: &Organism,
            dx: isize,
            dy: isize,
            spacing: isize,
        ) -> Option<(Organism, FxHashSet<Cell>)> {
            let mut offspring_cells = FxHashSet::default();
            let mut consumed_food = FxHashSet::default();
            let mut valid = true;

            // Check reproduction path and target area
            for cell in &parent.cells {
                // Check path cells
                for step in 1..=spacing {
                    let check_cell = Cell {
                        j: cell.j + dx * step,
                        i: cell.i + dy * step,
                        cell_type: CellType::Alive,
                    };

                    if let Some(existing) = self.cells.get(&check_cell) {
                        match existing.cell_type {
                            CellType::Alive => {
                                valid = false;
                                break;
                            }
                            CellType::Food => {
                                consumed_food.insert(check_cell);
                            }
                            _ => {}
                        }
                    }
                }

                // Create offspring cell
                let new_cell = Cell {
                    j: cell.j + dx * spacing,
                    i: cell.i + dy * spacing,
                    cell_type: CellType::Alive,
                };
                offspring_cells.insert(new_cell);
            }

            if valid && !consumed_food.is_empty() {
                Some((Organism::new(offspring_cells), consumed_food))
            } else {
                None
            }
        }

        /*

        fn tick(&mut self) {
            /*
            let mut adjacent_life = FxHashMap::default();
            // Counts the number of neighbours around a given cell, for each cell
            for cell in &self.cells {
                let _ = adjacent_life.entry(*cell).or_insert(0);

                for neighbor in Cell::neighbors(*cell) {
                    if neighbor.cell_type == CellType::Alive {
                        let amount = adjacent_life.entry(neighbor).or_insert(0);
                        *amount += 1;
                    }
                }
            }

            for (cell, amount) in &adjacent_life {
                match amount {
                    2 => {}
                    3 => {
                        let _ = self.cells.insert(*cell);
                    }
                    _ => {
                        if cell.cell_type == CellType::Alive {
                            let _ = self.cells.remove(cell);
                        }
                    }
                }
            }

            for mut organism in &self.organisms {
                for &cell in &organism.cells {
                    let _ = self.cells.remove(&cell);
                    let _ = self.cells.insert(Cell {
                        i: cell.i,
                        j: cell.j + 1,
                        cell_type: CellType::Alive,
                    });
                }
            }

            let mut new_organisms = FxHashSet::default();
            let mut new_cells = self.cells.clone(); // Keep all existing cells

            for organism in &self.organisms {
                let mut new_organism_cells = FxHashSet::default();

                // Remove old organism cells before updating positions
                for cell in &organism.cells {
                    new_cells.remove(cell); // Remove old position from grid
                }

                // Move the entire organism to the right
                for cell in &organism.cells {
                    let new_cell = Cell {
                        i: cell.i,                 // Maintain row position
                        j: cell.j + 1,             // Move right
                        cell_type: cell.cell_type, // Keep cell type
                    };

                    new_organism_cells.insert(new_cell);
                    new_cells.insert(new_cell); // Add new position to grid
                }

                // Store updated organism
                new_organisms.insert(Organism {
                    cells: new_organism_cells,
                });
            }

            // Update Life struct
            self.organisms = new_organisms;
            self.cells = new_cells; // Ensure proper rendering     }
            */

            self.move_organisms();
            self.handle_reproduction();
        }

        */

        pub fn iter(&self) -> impl Iterator<Item = &Cell> {
            self.cells.iter()
        }
    }

    // I am going to need to impelement a structure for an organism, and instead of cells were gonna have to make separate function to separate organisms
    impl std::iter::FromIterator<Cell> for Life {
        fn from_iter<I: IntoIterator<Item = Cell>>(iter: I) -> Self {
            let organisms: FxHashSet<Organism> = FxHashSet::default();
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
