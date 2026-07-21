"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import { api, Project } from "@/lib/api";
import ToastContainer, { ToastMessage } from "@/components/Toast";

interface DashboardStats {
  projects: number;
  datasets: number;
  experiments: number;
  production_models: number;
  top_model: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [stats, setStats] = useState<DashboardStats>({
    projects: 0,
    datasets: 0,
    experiments: 0,
    production_models: 0,
    top_model: "N/A"
  });
  
  // Create / Input States
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  // Search State
  const [searchQuery, setSearchQuery] = useState("");

  // Rename States
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [savingEdit, setSavingEdit] = useState(false);

  // Delete States
  const [deletingProject, setDeletingProject] = useState<Project | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Toast State
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  function addToast(text: string, type: "success" | "error" | "info" = "success") {
    const id = Math.random().toString();
    setToasts((prev) => [...prev, { id, type, text }]);
  }

  function removeToast(id: string) {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const [projRes, statsRes] = await Promise.all([
        api.get("/api/projects"),
        api.get("/api/projects/stats")
      ]);
      setProjects(projRes.data);
      setStats(statsRes.data);
    } catch (err: any) {
      addToast("Failed to load dashboard statistics", "error");
    } finally {
      setLoading(false);
    }
  }

  async function createProject(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    try {
      await api.post("/api/projects", { name, description });
      setName("");
      setDescription("");
      addToast("Project created successfully!", "success");
      await load();
    } catch (err: any) {
      addToast("Failed to create project", "error");
    } finally {
      setCreating(false);
    }
  }

  async function saveProjectRename(e: React.FormEvent) {
    e.preventDefault();
    if (!editingProject) return;
    setSavingEdit(true);
    try {
      await api.put(`/api/projects/${editingProject.id}`, {
        name: editName,
        description: editDescription
      });
      addToast("Project renamed successfully!", "success");
      setEditingProject(null);
      await load();
    } catch (err: any) {
      addToast("Failed to rename project", "error");
    } finally {
      setSavingEdit(false);
    }
  }

  async function deleteProjectConfirm() {
    if (!deletingProject) return;
    setDeleting(true);
    try {
      await api.delete(`/api/projects/${deletingProject.id}`);
      addToast("Project deleted successfully!", "success");
      setDeletingProject(null);
      await load();
    } catch (err: any) {
      addToast("Failed to delete project", "error");
    } finally {
      setDeleting(false);
    }
  }

  const filteredProjects = projects.filter(
    (p) =>
      p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (p.description && p.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="min-h-screen bg-paper text-ink selection:bg-accentSoft">
      <Navbar />
      <main className="max-w-5xl mx-auto px-6 py-10 space-y-10">
        
        {/* Statistics Dashboard Banner */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Total Projects" value={stats.projects} color="border-l-4 border-l-accent" />
          <StatCard label="Total Datasets" value={stats.datasets} color="border-l-4 border-l-blue-500" />
          <StatCard label="Total Models" value={stats.experiments} color="border-l-4 border-l-purple-500" />
          <StatCard label="Total Reports" value={stats.experiments} color="border-l-4 border-l-green-500" />
        </div>

        {/* Search & Actions Panel */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pt-4 border-t border-line">
          <div>
            <p className="label-eyebrow mb-1">workspace</p>
            <h1 className="font-display text-2xl font-semibold">Your projects</h1>
          </div>
          <div className="w-full md:max-w-xs">
            <input
              type="text"
              placeholder="Search projects..."
              className="input text-xs py-2 px-3"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        {/* Create Project Form */}
        <form onSubmit={createProject} className="card p-5 flex flex-col md:flex-row gap-3 md:items-end bg-paper/50">
          <div className="flex-1">
            <label className="label-eyebrow block mb-1">project name</label>
            <input
              className="input text-xs"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Sales Prediction"
            />
          </div>
          <div className="flex-1">
            <label className="label-eyebrow block mb-1">description / problem statement</label>
            <input
              className="input text-xs"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Forecast sales for next week"
            />
          </div>
          <button className="btn-primary shrink-0 text-xs py-2.5 px-4 font-semibold" disabled={creating}>
            {creating ? "creating..." : "new project"}
          </button>
        </form>

        {/* Projects List */}
        {loading ? (
          <p className="text-sm text-ink/60">loading projects…</p>
        ) : filteredProjects.length === 0 ? (
          <div className="border border-line rounded-lg p-10 text-center bg-paper/10">
            <p className="text-sm text-ink/50 font-medium">
              {searchQuery ? "No projects match your search criteria." : "No projects created yet. Add your first project above to get started!"}
            </p>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {filteredProjects.map((p) => (
              <div
                key={p.id}
                className="card p-5 flex flex-col justify-between hover:border-accent/40 transition-all group"
              >
                <div>
                  <div className="flex justify-between items-start mb-2">
                    <p className="label-eyebrow text-[10px]">project #{p.id} · {new Date(p.created_at).toLocaleDateString()}</p>
                    
                    {/* Action buttons */}
                    <div className="flex items-center gap-1.5 opacity-60 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => {
                          setEditingProject(p);
                          setEditName(p.name);
                          setEditDescription(p.description || "");
                        }}
                        className="text-[10px] text-ink/50 hover:text-accent font-semibold uppercase tracking-wider"
                      >
                        Rename
                      </button>
                      <span className="text-line">|</span>
                      <button
                        onClick={() => setDeletingProject(p)}
                        className="text-[10px] text-ink/50 hover:text-red-500 font-semibold uppercase tracking-wider"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  
                  <h2 className="font-display text-lg font-semibold text-ink group-hover:text-accent transition-colors">{p.name}</h2>
                  {p.description && (
                    <p className="text-xs text-ink/70 mt-2 bg-paper border border-line/45 p-2.5 rounded line-clamp-2">
                      {p.description}
                    </p>
                  )}
                </div>

                <div className="mt-4 pt-3 border-t border-line/45 flex justify-end">
                  <button
                    onClick={() => router.push(`/projects/${p.id}`)}
                    className="btn-secondary text-[11px] py-1 px-3 font-semibold uppercase tracking-wider"
                  >
                    Open Workspace →
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Rename Project Modal */}
      {editingProject && (
        <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <form onSubmit={saveProjectRename} className="bg-paper border border-line rounded-lg shadow-xl max-w-md w-full p-6 space-y-4">
            <h3 className="font-display font-semibold text-lg">Rename Project</h3>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] label-eyebrow block text-ink/60 mb-1">New Name</label>
                <input
                  type="text"
                  required
                  className="input text-xs"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                />
              </div>
              <div>
                <label className="text-[10px] label-eyebrow block text-ink/60 mb-1">New Description</label>
                <textarea
                  className="input text-xs h-20"
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                />
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setEditingProject(null)}
                className="btn-secondary text-xs py-1.5 px-3"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={savingEdit}
                className="btn-primary text-xs py-1.5 px-3"
              >
                {savingEdit ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deletingProject && (
        <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-paper border border-line rounded-lg shadow-xl max-w-sm w-full p-6 space-y-4">
            <h3 className="font-display font-semibold text-lg text-red-500">Delete Project?</h3>
            <p className="text-xs text-ink/70">
              Are you sure you want to delete project <strong>{deletingProject.name}</strong>? This will permanently delete all uploaded datasets and trained models inside this project.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setDeletingProject(null)}
                className="btn-secondary text-xs py-1.5 px-3"
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                onClick={deleteProjectConfirm}
                className="bg-red-500 hover:bg-red-600 text-paper rounded text-xs py-1.5 px-3 font-semibold transition-colors"
                disabled={deleting}
              >
                {deleting ? "Deleting..." : "Delete Permanently"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast Messages container */}
      <ToastContainer toasts={toasts} onClose={removeToast} />
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className={`card p-4 flex flex-col justify-between ${color} bg-paper/50`}>
      <p className="label-eyebrow text-ink/50 text-[10px] uppercase tracking-wider">{label}</p>
      <p className="font-display mt-2 font-bold text-2xl">
        {value}
      </p>
    </div>
  );
}
