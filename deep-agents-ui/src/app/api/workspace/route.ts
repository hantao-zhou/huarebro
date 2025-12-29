import { NextResponse } from "next/server";
import path from "path";
import { promises as fs } from "fs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type WorkspaceNode = {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  modified?: string;
  children?: WorkspaceNode[];
};

const workspaceDir = path.resolve(process.cwd(), "..", "workspace");

async function ensureWorkspaceDir() {
  await fs.mkdir(workspaceDir, { recursive: true });
}

function isWithinWorkspace(targetPath: string) {
  const relative = path.relative(workspaceDir, targetPath);
  return (
    relative === "" ||
    (!relative.startsWith("..") && !path.isAbsolute(relative))
  );
}

function toPosixPath(value: string) {
  return value.split(path.sep).join("/");
}

async function buildTree(
  currentPath: string,
  relativePath: string
): Promise<WorkspaceNode> {
  const stats = await fs.lstat(currentPath);
  const name = relativePath ? path.basename(currentPath) : "workspace";
  const nodePath = toPosixPath(relativePath);

  if (stats.isSymbolicLink()) {
    return {
      name,
      path: nodePath,
      type: "file",
      size: stats.size,
      modified: stats.mtime.toISOString(),
    };
  }

  if (stats.isDirectory()) {
    const entries = await fs.readdir(currentPath, { withFileTypes: true });
    const children = await Promise.all(
      entries.map(async (entry) => {
        const childRelative = path.join(relativePath, entry.name);
        const childAbsolute = path.join(currentPath, entry.name);
        if (!isWithinWorkspace(childAbsolute)) {
          return null;
        }
        return buildTree(childAbsolute, childRelative);
      })
    );

    const filtered = children.filter(
      (child): child is WorkspaceNode => child !== null
    );
    filtered.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === "directory" ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });

    return {
      name,
      path: nodePath,
      type: "directory",
      children: filtered,
    };
  }

  return {
    name,
    path: nodePath,
    type: "file",
    size: stats.size,
    modified: stats.mtime.toISOString(),
  };
}

export async function GET() {
  await ensureWorkspaceDir();
  const tree = await buildTree(workspaceDir, "");
  return NextResponse.json({ tree });
}

export async function POST(request: Request) {
  await ensureWorkspaceDir();
  const formData = await request.formData();
  const file = formData.get("file");

  if (!file || !(file instanceof File)) {
    return NextResponse.json(
      { error: "A file is required under the `file` field." },
      { status: 400 }
    );
  }

  const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
  if (!safeName) {
    return NextResponse.json(
      { error: "Invalid file name." },
      { status: 400 }
    );
  }

  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const filePath = path.join(workspaceDir, safeName);

  if (!isWithinWorkspace(filePath)) {
    return NextResponse.json({ error: "Invalid file path." }, { status: 400 });
  }

  await fs.writeFile(filePath, buffer);

  return NextResponse.json({
    file: {
      name: safeName,
      path: safeName,
      size: buffer.byteLength,
    },
  });
}
