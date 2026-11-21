import { z } from "zod";

export const loginValidator = z.object({
  email: z.email("Email is required and must be valid"),
  password: z.string("Password is required").trim().min(1, "Password is required"),
});

export type LoginValidatorType = z.infer<typeof loginValidator>;
