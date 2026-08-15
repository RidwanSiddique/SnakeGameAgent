/**
 * Level geometry for the browser.
 *
 * Imports shared/levels.json — the same file the Python trainer reads — rather
 * than a copy. A duplicate would drift, and a drifted board means the agent is
 * playing a level it was not trained on.
 */
import levelFile from '../../../shared/levels.json';
import { loadLevels, type Level, type LevelFile } from './levels.ts';

export const LEVELS: Map<number, Level> = loadLevels(levelFile as unknown as LevelFile);
export const LEVEL_LIST: Level[] = [...LEVELS.values()].sort((a, b) => a.id - b.id);
export const GRID = (levelFile as unknown as LevelFile).grid;
