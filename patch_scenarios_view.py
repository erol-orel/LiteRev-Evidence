#!/usr/bin/env python3
"""Remplace la section savedSearches dans ScenariosView par la section user_scenarios."""

with open("frontend/src/App.tsx", "r") as f:
    content = f.read()

# Trouver le début et la fin de la section à remplacer
start_marker = '      {/* ── Recherches sauvegardées ─────────────────────────────────────── */}'
# La section se termine par :
#       )}
#     </div>
#   );
# }
# On cherche la fin de la section savedSearches (avant </div>\n  );\n})

# Trouver la position de début
start_idx = content.index(start_marker)

# Trouver la fin : chercher le pattern de fermeture de la section
# La section se termine par "      )}\n    </div>\n  );\n}"
end_marker = '      )}\n    </div>\n  );\n}'
end_idx = content.index(end_marker, start_idx)
end_idx_full = end_idx + len(end_marker)

# Vérifier ce qu'on remplace
old_section = content[start_idx:end_idx_full]
print(f"Section trouvée: {len(old_section)} caractères")
print("Début:", repr(old_section[:100]))
print("Fin:", repr(old_section[-100:]))

new_section = '''      {/* ── Scénarios Utilisateur (recherches sauvegardées backend) ────────── */}
      {userScenarios.length > 0 && (
        <div className="mt-8 space-y-4">
          <div className="flex items-center gap-3">
            <RefreshCw size={18} className="text-gold-400" />
            <div>
              <h2 className="text-xl font-semibold text-white">Mes scénarios personnels</h2>
              <p className="text-xs text-forest-400 mt-0.5">
                Recherches sauvegardées — cliquez sur un scénario pour accéder à tous les onglets (corpus, PICO, screening, RAG, Evidence Brief...)
              </p>
            </div>
          </div>

          {/* Épinglés */}
          {userScenarios.filter(s => s.pinned).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-gold-400 uppercase tracking-widest">Épinglés</h3>
              {userScenarios.filter(s => s.pinned).map(s => (
                <div key={s.id} className="flex items-center gap-3 rounded-2xl border border-gold-400/30 bg-gold-500/5 px-4 py-3 hover:bg-gold-500/10 transition group">
                  <Activity size={14} className="text-gold-400 shrink-0" />
                  <div className="flex-1 min-w-0 cursor-pointer" onClick={() => setDetailScenarioId(s.id)}>
                    <p className="text-sm font-semibold text-white truncate group-hover:text-gold-300 transition">{s.title}</p>
                    <p className="text-xs text-white/40 truncate">
                      {s.mode === "semantic" ? "Sémantique" : "Booléen"} · {s.article_count} articles indexés
                      {s.created_at && <> · {new Date(s.created_at).toLocaleDateString("fr-CH")}</>}
                    </p>
                    {s.title !== s.query && <p className="text-xs text-white/25 truncate font-mono">{s.query}</p>}
                  </div>
                  <button
                    type="button"
                    onClick={() => setDetailScenarioId(s.id)}
                    className="shrink-0 rounded-xl bg-brand-500/20 border border-brand-500/30 px-3 py-1.5 text-xs text-brand-300 hover:bg-brand-500/30 transition"
                  >
                    Ouvrir
                  </button>
                  {onPopulateUserScenario && (
                    <button
                      type="button"
                      onClick={() => onPopulateUserScenario(s.id)}
                      disabled={populatingId === s.id}
                      className="shrink-0 rounded-xl border border-forest-500/30 px-2 py-1.5 text-xs text-forest-300 hover:bg-forest-500/10 transition disabled:opacity-50"
                      title="Ingérer des articles PubMed"
                    >
                      {populatingId === s.id ? <RotateCcw size={11} className="animate-spin" /> : <Zap size={11} />}
                    </button>
                  )}
                  {onReplaySearch && (
                    <button type="button" onClick={() => {
                      const saved = savedSearches.find(ss => ss.id === s.id);
                      if (saved) onReplaySearch(saved);
                    }} className="shrink-0 rounded-xl border border-white/10 px-2 py-1.5 text-xs text-white/40 hover:text-white hover:bg-white/10 transition" title="Relancer la recherche">&#8635;</button>
                  )}
                  {onTogglePin && (
                    <button type="button" onClick={() => onTogglePin(s.id)} className="shrink-0 rounded-xl border border-gold-400/20 px-2 py-1.5 text-xs text-gold-400 hover:bg-gold-500/10 transition" title="Désépingler">★</button>
                  )}
                  {onDeleteSearch && (
                    <button type="button" onClick={() => onDeleteSearch(s.id)} className="shrink-0 rounded-xl border border-white/10 px-2 py-1.5 text-xs text-white/30 hover:text-red-400 hover:border-red-500/20 transition" title="Supprimer">✕</button>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Non épinglés */}
          {userScenarios.filter(s => !s.pinned).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">Recherches récentes</h3>
              {userScenarios.filter(s => !s.pinned).map(s => (
                <div key={s.id} className="flex items-center gap-3 rounded-2xl border border-white/8 bg-white/3 px-4 py-3 hover:bg-white/8 transition group">
                  <BookOpen size={14} className="text-white/30 shrink-0" />
                  <div className="flex-1 min-w-0 cursor-pointer" onClick={() => setDetailScenarioId(s.id)}>
                    <p className="text-sm text-white/80 truncate group-hover:text-white transition">{s.title}</p>
                    <p className="text-xs text-white/30 truncate">
                      {s.mode === "semantic" ? "Sémantique" : "Booléen"} · {s.article_count} articles
                      {s.created_at && <> · {new Date(s.created_at).toLocaleDateString("fr-CH")}</>}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setDetailScenarioId(s.id)}
                    className="shrink-0 rounded-xl bg-white/5 border border-white/10 px-3 py-1.5 text-xs text-white/60 hover:text-white hover:bg-white/10 transition"
                  >
                    Ouvrir
                  </button>
                  {onPopulateUserScenario && (
                    <button
                      type="button"
                      onClick={() => onPopulateUserScenario(s.id)}
                      disabled={populatingId === s.id}
                      className="shrink-0 rounded-xl border border-forest-500/30 px-2 py-1.5 text-xs text-forest-300 hover:bg-forest-500/10 transition disabled:opacity-50"
                      title="Ingérer des articles PubMed"
                    >
                      {populatingId === s.id ? <RotateCcw size={11} className="animate-spin" /> : <Zap size={11} />}
                    </button>
                  )}
                  {onReplaySearch && (
                    <button type="button" onClick={() => {
                      const saved = savedSearches.find(ss => ss.id === s.id);
                      if (saved) onReplaySearch(saved);
                    }} className="shrink-0 rounded-xl border border-white/10 px-2 py-1.5 text-xs text-white/40 hover:text-white hover:bg-white/10 transition" title="Relancer">&#8635;</button>
                  )}
                  {onTogglePin && (
                    <button type="button" onClick={() => onTogglePin(s.id)} className="shrink-0 rounded-xl border border-white/10 px-2 py-1.5 text-xs text-white/30 hover:text-gold-400 hover:border-gold-400/30 transition" title="Épingler">☆</button>
                  )}
                  {onDeleteSearch && (
                    <button type="button" onClick={() => onDeleteSearch(s.id)} className="shrink-0 rounded-xl border border-white/10 px-2 py-1.5 text-xs text-white/30 hover:text-red-400 hover:border-red-500/20 transition" title="Supprimer">✕</button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}'''

new_content = content[:start_idx] + new_section + content[end_idx_full:]
print(f"\nNouveau contenu: {len(new_content)} caractères (original: {len(content)})")

with open("frontend/src/App.tsx", "w") as f:
    f.write(new_content)

print("✅ App.tsx mis à jour avec succès")
