import { revalidateLogic, useForm } from "@tanstack/react-form";
import { useMutation } from "@tanstack/react-query";
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { CircleAlertIcon, Loader2Icon, ShieldCheckIcon } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldSeparator,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";

import { getErrorMessage } from "@/lib/api-error";

import { login, meQueryOptions } from "@/api/auth";

import { loginValidator, type LoginValidatorType } from "@/validators/login.validator";

import { DEMO_ACCOUNTS, DEMO_PASSWORD } from "@/constants/demo-accounts";
import { QUERY_KEYS } from "@/constants/query-keys";

export const Route = createFileRoute("/login")({
  loader: async ({ context: { queryClient } }) => {
    const me = await queryClient.ensureQueryData(meQueryOptions).catch(() => null);
    if (me) throw redirect({ to: "/", replace: true });
  },
  component: LoginPage,
});

const defaultValues: LoginValidatorType = { email: "", password: "" };

function LoginPage() {
  const { queryClient } = Route.useRouteContext();
  const navigate = useNavigate();

  const loginMutation = useMutation({
    mutationFn: login,
    onSuccess: async (employee) => {
      // Login returns the same body as /auth/me, so seed the cache instead of
      // refetching it on the next page.
      queryClient.setQueryData(QUERY_KEYS.me, employee);
      await navigate({ to: "/" });
    },
  });

  const form = useForm({
    defaultValues,
    // Validate on submit, then on every change so errors clear as the user fixes them.
    validationLogic: revalidateLogic({
      mode: "submit",
      modeAfterSubmission: "change",
    }),
    validators: { onDynamic: loginValidator },
    onSubmit: ({ value }) => loginMutation.mutate(value),
  });

  const fillDemoAccount = (email: string) => {
    form.setFieldValue("email", email);
    form.setFieldValue("password", DEMO_PASSWORD);
  };

  return (
    <main className="flex min-h-svh items-center justify-center bg-muted/40 px-4 py-10">
      <div className="flex w-full max-w-sm flex-col gap-6">
        <div className="flex items-center justify-center gap-2 text-lg font-semibold">
          <span className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <ShieldCheckIcon className="size-5" />
          </span>
          Warden
        </div>

        <Card>
          <CardHeader className="text-center">
            <CardTitle className="text-xl">Welcome back</CardTitle>
            <CardDescription>Sign in to ask about policies, leave and your team</CardDescription>
          </CardHeader>

          <CardContent>
            <form
              id="login-form"
              noValidate
              onSubmit={(e) => {
                e.preventDefault();
                void form.handleSubmit();
              }}
            >
              <FieldGroup>
                {loginMutation.isError && (
                  <Alert variant="destructive">
                    <CircleAlertIcon />
                    <AlertDescription>{getErrorMessage(loginMutation.error)}</AlertDescription>
                  </Alert>
                )}

                <form.Field name="email">
                  {(field) => {
                    const isInvalid = field.state.meta.isTouched && !field.state.meta.isValid;
                    return (
                      <Field data-invalid={isInvalid}>
                        <FieldLabel htmlFor={field.name}>Email</FieldLabel>
                        <Input
                          id={field.name}
                          name={field.name}
                          type="email"
                          autoComplete="username"
                          placeholder="you@warden.demo"
                          value={field.state.value}
                          onBlur={field.handleBlur}
                          onChange={(e) => field.handleChange(e.target.value)}
                          aria-invalid={isInvalid}
                        />
                        {isInvalid && <FieldError errors={field.state.meta.errors} />}
                      </Field>
                    );
                  }}
                </form.Field>

                <form.Field name="password">
                  {(field) => {
                    const isInvalid = field.state.meta.isTouched && !field.state.meta.isValid;
                    return (
                      <Field data-invalid={isInvalid}>
                        <FieldLabel htmlFor={field.name}>Password</FieldLabel>
                        <Input
                          id={field.name}
                          name={field.name}
                          type="password"
                          autoComplete="current-password"
                          value={field.state.value}
                          onBlur={field.handleBlur}
                          onChange={(e) => field.handleChange(e.target.value)}
                          aria-invalid={isInvalid}
                        />
                        {isInvalid && <FieldError errors={field.state.meta.errors} />}
                      </Field>
                    );
                  }}
                </form.Field>

                <Field>
                  <Button type="submit" form="login-form" disabled={loginMutation.isPending}>
                    {loginMutation.isPending && <Loader2Icon className="animate-spin" />}
                    {loginMutation.isPending ? "Signing in…" : "Sign in"}
                  </Button>
                </Field>
              </FieldGroup>
            </form>
          </CardContent>

          <CardFooter className="flex-col items-stretch gap-4">
            <FieldSeparator className="*:data-[slot=field-separator-content]:bg-card">
              Demo accounts
            </FieldSeparator>
            <div className="grid grid-cols-3 gap-2">
              {DEMO_ACCOUNTS.map((account) => (
                <Button
                  key={account.email}
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => fillDemoAccount(account.email)}
                  disabled={loginMutation.isPending}
                >
                  {account.role}
                </Button>
              ))}
            </div>
            <FieldDescription className="text-center text-xs">
              {DEMO_ACCOUNTS.map((account) => account.email).join(" · ")}
              <br />
              Password: <span className="font-mono text-foreground">{DEMO_PASSWORD}</span>
            </FieldDescription>
          </CardFooter>
        </Card>
      </div>
    </main>
  );
}
