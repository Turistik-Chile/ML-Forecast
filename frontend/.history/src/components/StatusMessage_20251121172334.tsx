import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import type { StatusState } from "../types";
import { AlertCircle, CheckCircle2 } from "lucide-react";

type StatusMessageProps = StatusState;

const StatusMessage = ({ message, type }: StatusMessageProps) => {
  if (!message) return null;

  const isError = type === "error";

  return (
    <Alert className={isError ? "border-destructive/60 bg-destructive/10 text-destructive" : ""}>
      <AlertTitle className="flex items-center gap-2">
        {isError ? (
          <AlertCircle className="h-4 w-4" />
        ) : (
          <CheckCircle2 className="h-4 w-4 text-primary" />
        )}
        {isError ? "Ups, algo salió mal" : "Todo listo"}
      </AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
};

export default StatusMessage;