import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, real, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Facial features extracted from MediaPipe
export const facialFeaturesSchema = z.object({
  mouthOpen: z.number().min(0).max(1),
  eyeClosureAvg: z.number().min(0).max(1),
  browFurrowAvg: z.number().min(0).max(1),
  headTiltVar: z.number().min(0).max(1),
  microMovementVar: z.number().min(0).max(1),
});

// Caregiver assessment inputs
export const caregiverInputSchema = z.object({
  grimace: z.number().min(0).max(5),
  breathing: z.number().min(0).max(5),
  restlessness: z.number().min(0).max(5),
  gestures: z.array(z.enum(["clench", "point", "shake"])).default([]),
});

// Complete pain assessment record
export const painAssessments = pgTable("pain_assessments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
  facialFeatures: jsonb("facial_features").$type<z.infer<typeof facialFeaturesSchema>>().notNull(),
  caregiverInputs: jsonb("caregiver_inputs").$type<z.infer<typeof caregiverInputSchema>>().notNull(),
  painScore: real("pain_score").notNull(),
  confidence: real("confidence").notNull(),
  topContributors: jsonb("top_contributors").$type<string[]>().notNull(),
});

export const insertPainAssessmentSchema = createInsertSchema(painAssessments).omit({
  id: true,
  timestamp: true,
});

export type FacialFeatures = z.infer<typeof facialFeaturesSchema>;
export type CaregiverInput = z.infer<typeof caregiverInputSchema>;
export type PainAssessment = typeof painAssessments.$inferSelect;
export type InsertPainAssessment = z.infer<typeof insertPainAssessmentSchema>;

// Users table (kept for future use)
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
