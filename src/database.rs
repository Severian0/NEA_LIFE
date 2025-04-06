use std::collections::{HashMap, HashSet};

use crate::grid::{Cell, CellType, Organism};

use rusqlite::{params, Connection, Result};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]

pub struct WorldState {
    pub organisms: Vec<Organism>, // Note: Organism no longer needs an id field

    pub free_cells: Vec<Cell>,

    pub version: usize,
}

impl Organism {
    pub fn is_valid(&self) -> bool {
        !self.cells.is_empty() && self.id.unwrap() > 0
    }
}

impl WorldState {
    pub fn validate(&self) -> Result<(), String> {
        // Check for duplicate organism IDs

        let mut ids = HashSet::new();

        for org in &self.organisms {
            if !org.is_valid() {
                return Err(format!("Invalid organism: {:?}", org));
            }

            if !ids.insert(org.id) {
                return Err(format!("Duplicate organism ID: {}", org.id.unwrap()));
            }
        }

        // Check for cell conflicts

        let mut cell_positions = HashSet::new();

        for org in &self.organisms {
            for cell in &org.cells {
                if !cell_positions.insert((cell.i, cell.j)) {
                    return Err(format!("Duplicate cell position: ({}, {})", cell.i, cell.j));
                }
            }
        }

        for cell in &self.free_cells {
            if !cell_positions.insert((cell.i, cell.j)) {
                return Err(format!("Cell conflict at: ({}, {})", cell.i, cell.j));
            }
        }

        Ok(())
    }
}

pub struct WorldDatabase {
    conn: Connection,
}

impl WorldDatabase {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute_batch(
            "PRAGMA foreign_keys = ON;

             BEGIN;

             

             CREATE TABLE IF NOT EXISTS organisms (

                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Changed to auto-increment

                energy INTEGER NOT NULL,

                able_to_move BOOLEAN NOT NULL

             );

             

             CREATE TABLE IF NOT EXISTS cells (

                i INTEGER NOT NULL,

                j INTEGER NOT NULL,

                cell_type TEXT NOT NULL,

                organism_id INTEGER,

                PRIMARY KEY (i, j),

                FOREIGN KEY (organism_id) REFERENCES organisms(id) ON DELETE CASCADE

             );

             

             CREATE TABLE IF NOT EXISTS world_states (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                version INTEGER NOT NULL,

                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP

             );

             

             COMMIT;",
        )?;

        Ok(Self { conn })
    }

    pub fn save_state(&mut self, state: &WorldState) -> Result<()> {
        let tx = self.conn.transaction()?;

        // Clear existing data in correct order

        tx.execute("DELETE FROM cells", [])?;

        tx.execute("DELETE FROM organisms", [])?;

        tx.execute("DELETE FROM world_states", [])?;

        // Save version info

        tx.execute(
            "INSERT INTO world_states (version) VALUES (?1)",
            params![state.version],
        )?;

        // Save organisms and get their generated IDs

        let mut organism_ids = Vec::new();

        for organism in &state.organisms {
            tx.execute(
                "INSERT INTO organisms (energy, able_to_move) VALUES (?1, ?2)",
                params![organism.energy, organism.able_to_move],
            )?;

            let id = tx.last_insert_rowid() as usize;

            organism_ids.push(id);
        }

        // Track the newest cell at each position

        let mut newest_cells = std::collections::HashMap::new();

        // Process organism cells (in order they appear in organisms)

        for (org_idx, organism) in state.organisms.iter().enumerate() {
            let organism_id = organism_ids[org_idx];

            for cell in &organism.cells {
                newest_cells.insert((cell.i, cell.j), (cell.clone(), Some(organism_id)));
            }
        }

        // Process free cells (these will overwrite organism cells if same position)

        for cell in &state.free_cells {
            newest_cells.insert((cell.i, cell.j), (cell.clone(), None));
        }

        // Save only the newest cells (last one inserted at each position wins)

        for ((i, j), (cell, organism_id)) in newest_cells {
            tx.execute(
                "INSERT INTO cells (i, j, cell_type, organism_id) VALUES (?1, ?2, ?3, ?4)",
                params![
                    i,
                    j,
                    serde_json::to_string(&cell.cell_type)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    organism_id
                ],
            )?;
        }

        tx.commit()
    }

    pub fn load_latest_state(&self) -> Result<WorldState> {
        let version = self
            .conn
            .query_row(
                "SELECT version FROM world_states ORDER BY timestamp DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Load all cells first (both organism and free cells)

        let mut stmt = self
            .conn
            .prepare("SELECT i, j, cell_type, organism_id FROM cells")?;

        let all_cells = stmt
            .query_map([], |row| {
                let cell_type_str: String = row.get(2)?;

                let cell_type = serde_json::from_str(&cell_type_str).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;

                Ok((
                    Cell {
                        i: row.get(0)?,

                        j: row.get(1)?,

                        cell_type,
                    },
                    row.get::<_, Option<usize>>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>>>()?;

        // Group cells by organism

        let mut organism_cells: HashMap<usize, Vec<Cell>> = HashMap::new();

        let mut free_cells = Vec::new();

        for (cell, organism_id) in all_cells {
            if let Some(org_id) = organism_id {
                organism_cells.entry(org_id).or_default().push(cell);
            } else {
                free_cells.push(cell);
            }
        }

        // Load organisms with their cells

        let mut stmt = self
            .conn
            .prepare("SELECT id, energy, able_to_move FROM organisms")?;

        let organisms = stmt
            .query_map([], |row| {
                let id: usize = row.get(0)?;

                let energy: usize = row.get(1)?;

                let able_to_move: bool = row.get(2)?;

                let cells = organism_cells.remove(&id).unwrap_or_default();

                Ok(Organism {
                    id: Some(id),

                    cells,

                    energy,

                    able_to_move,
                })
            })?
            .collect::<Result<Vec<_>>>()?;

        Ok(WorldState {
            organisms,

            free_cells,

            version,
        })
    }
}
