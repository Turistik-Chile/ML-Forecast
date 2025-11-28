import type { StatusState } from "../types";

type StatusMessageProps = StatusState;

const StatusMessage = ({ message, type }: StatusMessageProps) => {
  if (!message) {
    return null;
  }

  return (
    <div className={`status ${type === "error" ? "error" : ""}`}>{message}</div>
  );
};

export default StatusMessage;
