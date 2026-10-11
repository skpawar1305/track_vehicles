import { writable } from "svelte/store";
import type { DayCounts, DaysMap, DayEvent } from "$lib/types";

export type { DayCounts, DaysMap, DayEvent };

export const days = writable<DaysMap>({});
export const todayCounts = writable<DayCounts>({ in: 0, out: 0 });
export const edge = writable<{ stats: Record<string, unknown> | null; age: number | null }>({ stats: null, age: null });
export const cam2 = writable(false);
export const liveOk = writable(false);
export const setupOpen = writable(false);
