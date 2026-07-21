import React, { useEffect } from "react";

export interface ToastMessage {
  id: string;
  type: "success" | "error" | "info";
  text: string;
}

interface ToastProps {
  toasts: ToastMessage[];
  onClose: (id: string) => void;
}

export default function ToastContainer({ toasts, onClose }: ToastProps) {
  return (
    <div className="fixed bottom-5 right-5 z-50 space-y-2 pointer-events-none max-w-sm w-full">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onClose={onClose} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onClose }: { toast: ToastMessage; onClose: (id: string) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose(toast.id);
    }, 4000);
    return () => clearTimeout(timer);
  }, [toast.id, onClose]);

  return (
    <div
      className={`pointer-events-auto p-4 rounded-lg shadow-lg border flex justify-between items-center gap-3 transition-all duration-300 transform translate-x-0 ${
        toast.type === "success"
          ? "bg-green-50 border-green-200 text-green-800"
          : toast.type === "error"
          ? "bg-red-50 border-red-200 text-red-800"
          : "bg-blue-50 border-blue-200 text-blue-800"
      }`}
    >
      <span className="text-xs font-semibold font-sans">{toast.text}</span>
      <button
        onClick={() => onClose(toast.id)}
        className="text-ink/30 hover:text-ink/80 text-[10px] font-bold tracking-wider shrink-0 transition-colors"
      >
        ✕
      </button>
    </div>
  );
}
