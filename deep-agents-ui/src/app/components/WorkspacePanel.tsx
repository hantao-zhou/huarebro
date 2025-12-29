"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FileText,
  Folder,
  Loader2,
  RefreshCw,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type WorkspaceNode = {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  modified?: string;
  children?: WorkspaceNode[];
};

const ROOT_KEY = "__workspace_root__";

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "0 B";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${
    units[exponent]
  }`;
}

export function WorkspacePanel() {
  const [open, setOpen] = useState(false);
  const [tree, setTree] = useState<WorkspaceNode | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [downloading, setDownloading] = useState<Set<string>>(
    () => new Set()
  );
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set([ROOT_KEY])
  );
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchTree = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/workspace", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Unable to load workspace files");
      }
      const data = await response.json();
      setTree(data?.tree ?? null);
    } catch (error: any) {
      toast.error(error?.message || "Failed to load workspace files");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      fetchTree();
    }
  }, [open, fetchTree]);

  const toggleExpanded = useCallback((key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  const handleUpload = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList?.length) return;
      setIsUploading(true);
      try {
        for (const file of Array.from(fileList)) {
          const formData = new FormData();
          formData.append("file", file);
          const response = await fetch("/api/workspace", {
            method: "POST",
            body: formData,
          });
          if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.error || "Failed to upload file");
          }
        }
        toast.success(
          `Uploaded ${fileList.length} file${fileList.length === 1 ? "" : "s"}`
        );
        await fetchTree();
      } catch (error: any) {
        toast.error(error?.message || "Failed to upload file");
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [fetchTree]
  );

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleDragOver = useCallback(
    (event: React.DragEvent<HTMLButtonElement>) => {
      event.preventDefault();
      if (isUploading) return;
      event.dataTransfer.dropEffect = "copy";
      setIsDragging(true);
    },
    [isUploading]
  );

  const handleDragLeave = useCallback(
    (event: React.DragEvent<HTMLButtonElement>) => {
      const related = event.relatedTarget;
      if (related && event.currentTarget.contains(related as Node)) {
        return;
      }
      setIsDragging(false);
    },
    []
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLButtonElement>) => {
      event.preventDefault();
      setIsDragging(false);
      if (isUploading) return;
      handleUpload(event.dataTransfer.files);
    },
    [handleUpload, isUploading]
  );

  const treeStats = useMemo(() => {
    if (!tree) {
      return { files: 0, directories: 0 };
    }

    const walk = (node: WorkspaceNode, isRoot: boolean) => {
      if (node.type === "file") {
        return { files: 1, directories: 0 };
      }

      let files = 0;
      let directories = isRoot ? 0 : 1;

      node.children?.forEach((child) => {
        const counts = walk(child, false);
        files += counts.files;
        directories += counts.directories;
      });

      return { files, directories };
    };

    return walk(tree, true);
  }, [tree]);

  const treeSummary = tree
    ? `${treeStats.files} file${treeStats.files === 1 ? "" : "s"} - ${
        treeStats.directories
      } folder${treeStats.directories === 1 ? "" : "s"}`
    : isLoading
      ? "Loading workspace tree..."
      : "Workspace data unavailable.";

  const expandAll = useCallback(() => {
    if (!tree) return;
    const next = new Set<string>();
    const visit = (node: WorkspaceNode) => {
      if (node.type === "directory") {
        next.add(node.path || ROOT_KEY);
        node.children?.forEach(visit);
      }
    };
    visit(tree);
    setExpanded(next);
  }, [tree]);

  const collapseAll = useCallback(() => {
    setExpanded(new Set([ROOT_KEY]));
  }, []);

  const updateDownloadState = useCallback(
    (nodeKey: string, isActive: boolean) => {
      setDownloading((prev) => {
        const next = new Set(prev);
        if (isActive) {
          next.add(nodeKey);
        } else {
          next.delete(nodeKey);
        }
        return next;
      });
    },
    []
  );

  const handleDownload = useCallback(
    async (node: WorkspaceNode) => {
      if (!node.path) {
        toast.error("Unable to download this file.");
        return;
      }
      const nodeKey = node.path || ROOT_KEY;
      updateDownloadState(nodeKey, true);
      try {
        const params = new URLSearchParams({ path: node.path });
        const response = await fetch(
          `/api/workspace/download?${params.toString()}`
        );
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || "Failed to download file");
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = node.name || "download";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } catch (error: any) {
        toast.error(error?.message || "Failed to download file");
      } finally {
        updateDownloadState(nodeKey, false);
      }
    },
    [updateDownloadState]
  );

  const renderNode = useCallback(
    (node: WorkspaceNode, depth: number) => {
      const nodeKey = node.path || ROOT_KEY;
      const displayPath = node.path ? `workspace/${node.path}` : "workspace";
      const isDirectory = node.type === "directory";
      const isExpanded = expanded.has(nodeKey);
      const isDownloading = downloading.has(nodeKey);
      const indent = depth * 16 + 8;

      const rowContent = (
        <>
          {isDirectory ? (
            isExpanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )
          ) : (
            <span className="h-4 w-4" />
          )}
          {isDirectory ? (
            <Folder className="h-4 w-4 text-muted-foreground" />
          ) : (
            <FileText className="h-4 w-4 text-muted-foreground" />
          )}
          <span
            className="truncate text-sm text-foreground"
            title={displayPath}
          >
            {node.name}
          </span>
          {node.type === "file" && (
            <div className="ml-auto flex items-center gap-2">
              {node.size != null && (
                <span className="text-xs text-muted-foreground">
                  {formatBytes(node.size)}
                </span>
              )}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => handleDownload(node)}
                disabled={isDownloading}
                aria-label={`Download ${node.name}`}
                title={`Download ${node.name}`}
              >
                {isDownloading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
              </Button>
            </div>
          )}
        </>
      );

      return (
        <div key={nodeKey}>
          {isDirectory ? (
            <button
              type="button"
              onClick={() => toggleExpanded(nodeKey)}
              className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left hover:bg-accent/60"
              style={{ paddingLeft: indent }}
              aria-expanded={isExpanded}
            >
              {rowContent}
            </button>
          ) : (
            <div
              className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left"
              style={{ paddingLeft: indent }}
            >
              {rowContent}
            </div>
          )}
          {isDirectory &&
            isExpanded &&
            node.children &&
            node.children.map((child) => renderNode(child, depth + 1))}
        </div>
      );
    },
    [downloading, expanded, handleDownload, toggleExpanded]
  );

  const treeContent = (() => {
    if (isLoading) {
      return (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Loading workspace...
        </div>
      );
    }

    if (!tree) {
      return (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          Workspace data unavailable.
        </div>
      );
    }

    if (!tree.children || tree.children.length === 0) {
      return (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          No files found in /workspace.
        </div>
      );
    }

    return renderNode(tree, 0);
  })();

  return (
    <Dialog
      open={open}
      onOpenChange={setOpen}
    >
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
        >
          Workspace
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90vh] max-w-4xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>Workspace files</DialogTitle>
          <DialogDescription>
            Upload files to /workspace and browse the directory tree.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Upload
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Drop files into /workspace or browse to upload.
                </p>
              </div>
              {isUploading && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading...
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={openFilePicker}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              disabled={isUploading}
              className={cn(
                "mt-4 flex min-h-[160px] w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-4 py-6 text-center transition-colors",
                isDragging
                  ? "border-primary bg-primary/5"
                  : "border-border/60 bg-muted/30",
                isUploading
                  ? "cursor-not-allowed opacity-60"
                  : "hover:border-primary/60 hover:bg-primary/5"
              )}
            >
              <Upload className="h-6 w-6 text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">
                Drop files here
              </span>
              <span className="text-xs text-muted-foreground">
                or click to browse
              </span>
            </button>
            <p className="mt-3 text-xs text-muted-foreground">
              Files are stored under /workspace for agent access.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="sr-only"
              onChange={(event) => handleUpload(event.target.files)}
            />
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Workspace tree
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  {treeSummary}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={expandAll}
                  disabled={!tree?.children?.length}
                >
                  Expand all
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={collapseAll}
                  disabled={!tree?.children?.length}
                >
                  Collapse
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={fetchTree}
                  disabled={isLoading}
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </Button>
              </div>
            </div>
            <ScrollArea className="mt-3 h-[360px] pr-2">
              {treeContent}
            </ScrollArea>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
