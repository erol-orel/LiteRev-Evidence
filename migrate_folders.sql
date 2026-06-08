ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS folder_id TEXT;
CREATE TABLE IF NOT EXISTS user_scenario_folders (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  color TEXT NOT NULL DEFAULT '#6366f1',
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW()
);
SELECT 'Migration OK';
