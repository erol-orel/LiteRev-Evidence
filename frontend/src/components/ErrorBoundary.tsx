import React from "react";
import { tStandalone } from "../i18n/LanguageProvider";

interface Props {
  children: React.ReactNode;
  /** When this value changes, a tripped boundary auto-resets (e.g. on tab/scenario change). */
  resetKey?: unknown;
  /** Optional custom fallback. */
  fallback?: React.ReactNode;
  /** Called when the user clicks the retry button. */
  onReset?: () => void;
  /** i18n key for the section label, resolved and shown in the fallback. */
  label?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Limite d'erreur React. Sans cela, n'importe quelle exception de rendu démonte
 * tout l'arbre React → écran noir. Ici on isole le crash et on affiche un message
 * localisé, le reste de l'application reste utilisable.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Visible en console pour le diagnostic ; n'interrompt pas l'app.
    console.error("[ErrorBoundary]", this.props.label ?? "", error, info?.componentStack);
  }

  componentDidUpdate(prev: Props) {
    if (this.state.hasError && prev.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, error: null });
    }
  }

  private reset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  render() {
    if (!this.state.hasError) return this.props.children;
    if (this.props.fallback) return this.props.fallback;
    return (
      <div className="rounded-2xl border border-red-500/20 bg-red-500/5 px-4 py-4 text-sm">
        <p className="font-semibold text-red-300 mb-1">
          {tStandalone("errorBoundary.title")}{this.props.label ? ` ${tStandalone("errorBoundary.ofSection")} « ${tStandalone(this.props.label)} »` : ""}.
        </p>
        <p className="text-xs text-red-200/70 mb-3 break-words">
          {this.state.error?.message || tStandalone("errorBoundary.unknownError")}
        </p>
        <button
          onClick={this.reset}
          className="rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-1.5 text-xs text-red-200 hover:bg-red-500/20 transition"
        >
          {tStandalone("errorBoundary.retry")}
        </button>
      </div>
    );
  }
}
