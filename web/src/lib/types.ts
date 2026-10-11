export type DayCounts = { in: number; out: number };
export type DaysMap = Record<string, DayCounts>;
export type DayEvent = {
  id: string;
  label: string;
  dir: string;
  track: number | null;
  ts: string;
  sub: boolean;
  ign: boolean;
};
