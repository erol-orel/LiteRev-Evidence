// Suivre un travail que le SERVEUR mène (recherche, clustering, brief, entraînement,
// lot d'enrichissement...). Chaque écran avait sa boucle, et chacune s'arrêtait à sa
// façon : au bout de 15 minutes, de 40 essais, à la première erreur réseau, et disait
// alors « échec » ou « terminé » pendant que le serveur continuait. Les règles, ici
// une fois :
//   - on sonde TANT QUE le serveur dit « en cours », sans plafond ;
//   - après quelques minutes, plus lentement (toutes les 15 s) ;
//   - une erreur passagère ne l'arrête pas : seules plusieurs erreurs DE SUITE le font ;
//   - quand l'écran s'arrête pour une autre raison que la fin du travail, l'issue le dit
//     (`lost`), pour que l'écran annonce « le travail continue sur le serveur » et non
//     un échec.

export interface PollSchedule {
  /** Intervalle tant que le travail est récent (ms). */
  intervalMs: number;
  /** Au-delà de ce temps, on sonde toutes les `slowIntervalMs` (5 minutes par défaut). */
  slowAfterMs?: number;
  /** 15 s par défaut ; jamais plus rapide que `intervalMs`. */
  slowIntervalMs?: number;
}

export const POLL_SLOW_AFTER_MS = 5 * 60_000;
export const POLL_SLOW_INTERVAL_MS = 15_000;
/** Lectures en échec consécutives tolérées avant d'abandonner le suivi. */
export const POLL_MAX_ERRORS = 5;

/** Le délai avant la prochaine lecture, selon le temps écoulé depuis le début. Pur. */
export function pollDelay(elapsedMs: number, s: PollSchedule): number {
  const slowAfter = s.slowAfterMs ?? POLL_SLOW_AFTER_MS;
  const slow = Math.max(s.intervalMs, s.slowIntervalMs ?? POLL_SLOW_INTERVAL_MS);
  return elapsedMs >= slowAfter ? slow : s.intervalMs;
}

export type PollOutcome<T> =
  /** Le serveur ne dit plus « en cours » : `status` est sa dernière réponse. */
  | { kind: "done"; status: T }
  /** Trop de lectures en échec de suite : le travail continue PEUT-ÊTRE sur le serveur. */
  | { kind: "lost"; error: unknown }
  /** L'écran a cessé d'écouter (démontage, nouvelle demande). */
  | { kind: "cancelled" };

export interface PollOptions<T> extends PollSchedule {
  /** Une lecture de l'état. */
  read: () => Promise<T>;
  /** Vrai tant que le serveur dit que le travail tourne. */
  running: (status: T) => boolean;
  /** Chaque lecture réussie, pour afficher la progression. */
  onStatus?: (status: T) => void;
  /** Délai avant la première lecture (par défaut `intervalMs`). */
  firstDelayMs?: number;
  /** Échecs consécutifs tolérés (POLL_MAX_ERRORS par défaut). */
  maxErrors?: number;
  /** Horloge, remplaçable par les tests. */
  now?: () => number;
}

export interface PollHandle<T> {
  promise: Promise<PollOutcome<T>>;
  cancel: () => void;
}

export function pollJob<T>(o: PollOptions<T>): PollHandle<T> {
  const now = o.now ?? (() => Date.now());
  const start = now();
  const maxErrors = Math.max(1, o.maxErrors ?? POLL_MAX_ERRORS);
  let finished = false;
  let errors = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let settle: (v: PollOutcome<T>) => void = () => {};
  const promise = new Promise<PollOutcome<T>>((resolve) => { settle = resolve; });

  const finish = (v: PollOutcome<T>) => {
    if (finished) return;
    finished = true;
    if (timer) clearTimeout(timer);
    timer = null;
    settle(v);
  };
  const next = () => { timer = setTimeout(tick, pollDelay(now() - start, o)); };

  async function tick() {
    timer = null;
    if (finished) return;
    let status: T;
    try {
      status = await o.read();
    } catch (error) {
      if (finished) return;
      errors += 1;
      if (errors >= maxErrors) finish({ kind: "lost", error });
      else next();
      return;
    }
    if (finished) return;
    errors = 0;
    try {
      o.onStatus?.(status);
    } catch (e) {
      console.warn("pollJob onStatus", e);
    }
    if (o.running(status)) next();
    else finish({ kind: "done", status });
  }

  timer = setTimeout(tick, o.firstDelayMs ?? o.intervalMs);
  return { promise, cancel: () => finish({ kind: "cancelled" }) };
}
