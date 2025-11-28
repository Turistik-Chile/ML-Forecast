import * as React from "react";
import { Button } from "./button";
import { Card, CardTitle, CardContent } from "./card";
import { cn } from "../../lib/utils";

type ConfirmDialogProps = {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onClose: () => void;
};

const ConfirmDialog = ({
  open,
  title,
  description,
  confirmLabel = "Confirmar",
  cancelLabel = "Cancelar",
  onConfirm,
  onClose,
}: ConfirmDialogProps) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <Card className="w-full max-w-sm space-y-4 border">
        <CardContent className="space-y-3">
          <CardTitle className="text-lg">{title}</CardTitle>
          {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
        </CardContent>
        <div className="flex justify-end gap-2 px-6 pb-6 pt-2">
          <Button variant="outline" onClick={onClose}>
            {cancelLabel}
          </Button>
          <Button onClick={onConfirm}>{confirmLabel}</Button>
        </div>
      </Card>
    </div>
  );
};

export { ConfirmDialog };
