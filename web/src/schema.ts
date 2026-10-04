import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";

export const events = sqliteTable("events", {
  id: text("id").primaryKey(),
  trackId: integer("track_id"),
  classId: integer("class_id"),
  label: text("label"),
  confidence: real("confidence"),
  direction: text("direction"),
  crossedAt: text("crossed_at"),
  bbox: text("bbox"),
  line: text("line"),
  imagePath: text("image_path"),
  thumbPath: text("thumb_path"),
  createdAt: text("created_at").notNull(),
});
