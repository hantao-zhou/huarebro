"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import {
  type Message,
  type Assistant,
  type Checkpoint,
} from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";
import type { UseStreamThread } from "@langchain/langgraph-sdk/react";
import type { FileData, FileInputMap, FileMap, TodoItem } from "@/app/types/types";
import { useClient } from "@/providers/ClientProvider";
import { useQueryState } from "nuqs";

export type StateType = {
  messages: Message[];
  todos: TodoItem[];
  files: FileInputMap;
  email?: {
    id?: string;
    subject?: string;
    page_content?: string;
  };
  ui?: any;
};

const normalizeFileValue = (
  value: FileData | string,
  existing?: FileData
): FileData => {
  const now = new Date().toISOString();
  if (typeof value === "string") {
    return {
      ...(existing ?? {}),
      content: value.split("\n"),
      created_at: existing?.created_at ?? now,
      modified_at: now,
    };
  }

  if (value && typeof value === "object") {
    const contentSource = (value as { content?: unknown }).content;
    const content = Array.isArray(contentSource)
      ? contentSource.map((line) => String(line))
      : typeof contentSource === "string"
        ? contentSource.split("\n")
        : [];
    const createdAt =
      typeof (value as { created_at?: unknown }).created_at === "string"
        ? (value as { created_at: string }).created_at
        : existing?.created_at ?? now;
    const modifiedAt =
      typeof (value as { modified_at?: unknown }).modified_at === "string"
        ? (value as { modified_at: string }).modified_at
        : existing?.modified_at ?? now;
    return {
      ...(existing ?? {}),
      ...value,
      content,
      created_at: createdAt,
      modified_at: modifiedAt,
    };
  }

  return { content: [], created_at: now, modified_at: now };
};

const normalizeFiles = (
  files: FileInputMap | null | undefined,
  existing?: FileMap
): FileMap => {
  const normalized: FileMap = {};
  if (!files || typeof files !== "object") return normalized;
  for (const [path, value] of Object.entries(files)) {
    normalized[path] = normalizeFileValue(value, existing?.[path]);
  }
  return normalized;
};

export function useChat({
  activeAssistant,
  onHistoryRevalidate,
  thread,
}: {
  activeAssistant: Assistant | null;
  onHistoryRevalidate?: () => void;
  thread?: UseStreamThread<StateType>;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const client = useClient();
  const recursionLimit = 999;

  const stream = useStream<StateType>({
    assistantId: activeAssistant?.assistant_id || "",
    client: client ?? undefined,
    reconnectOnMount: true,
    fetchStateHistory: true,
    threadId: threadId ?? null,
    onThreadId: setThreadId,
    defaultHeaders: { "x-auth-scheme": "langsmith" },
    // Revalidate thread list when stream finishes, errors, or creates new thread
    onFinish: onHistoryRevalidate,
    onError: onHistoryRevalidate,
    onCreated: onHistoryRevalidate,
    experimental_thread: thread,
  });

  const [localFiles, setLocalFiles] = useState<FileMap>(() =>
    normalizeFiles(stream.values.files)
  );
  const localFilesRef = useRef(localFiles);
  const pendingFilesRef = useRef<FileMap | null>(null);
  const latestStreamFilesRef = useRef(stream.values.files);

  useEffect(() => {
    localFilesRef.current = localFiles;
  }, [localFiles]);

  useEffect(() => {
    pendingFilesRef.current = null;
  }, [threadId]);

  const streamFiles = stream.values.files;
  useEffect(() => {
    latestStreamFilesRef.current = streamFiles;
    if (pendingFilesRef.current) return;
    setLocalFiles(normalizeFiles(streamFiles, localFilesRef.current));
  }, [streamFiles]);

  const syncFilesFromStream = useCallback(() => {
    const latestFiles = latestStreamFilesRef.current;
    setLocalFiles(normalizeFiles(latestFiles, localFilesRef.current));
  }, []);

  const sendMessage = useCallback(
    (content: string) => {
      const newMessage: Message = { id: uuidv4(), type: "human", content };
      const pendingFiles = pendingFilesRef.current;
      const submitValues = pendingFiles
        ? { messages: [newMessage], files: pendingFiles }
        : { messages: [newMessage] };
      const submitPromise = stream.submit(submitValues, {
        optimisticValues: (prev) => ({
          messages: [...(prev.messages ?? []), newMessage],
          ...(pendingFiles ? { files: pendingFiles } : {}),
        }),
        config: { ...(activeAssistant?.config ?? {}), recursion_limit: recursionLimit },
      });
      if (pendingFiles) {
        const pendingSnapshot = pendingFiles;
        void submitPromise
          .finally(() => {
            if (pendingFilesRef.current !== pendingSnapshot) {
              return;
            }
            pendingFilesRef.current = null;
            syncFilesFromStream();
          });
      }
      // Update thread list immediately when sending a message
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate, recursionLimit, syncFilesFromStream]
  );

  const runSingleStep = useCallback(
    (
      messages: Message[],
      checkpoint?: Checkpoint,
      isRerunningSubagent?: boolean,
      optimisticMessages?: Message[]
    ) => {
      if (checkpoint) {
        stream.submit(undefined, {
          ...(optimisticMessages
            ? { optimisticValues: { messages: optimisticMessages } }
            : {}),
          config: { ...(activeAssistant?.config ?? {}), recursion_limit: recursionLimit },
          checkpoint: checkpoint,
          ...(isRerunningSubagent
            ? { interruptAfter: ["tools"] }
            : { interruptBefore: ["tools"] }),
        });
      } else {
        stream.submit(
          { messages },
          {
            config: { ...(activeAssistant?.config ?? {}), recursion_limit: recursionLimit },
            interruptBefore: ["tools"],
          }
        );
      }
    },
    [stream, activeAssistant?.config, recursionLimit]
  );

  const setFiles = useCallback(
    async (files: FileInputMap) => {
      const normalized = normalizeFiles(files, localFilesRef.current);
      pendingFilesRef.current = normalized;
      setLocalFiles(normalized);
      if (!threadId) return;
      // TODO: missing a way how to revalidate the internal state
      // I think we do want to have the ability to externally manage the state
      await client.threads.updateState(threadId, { values: { files: normalized } });
    },
    [client, threadId]
  );

  const continueStream = useCallback(
    (hasTaskToolCall?: boolean) => {
      stream.submit(undefined, {
        config: {
          ...(activeAssistant?.config || {}),
          recursion_limit: recursionLimit,
        },
        ...(hasTaskToolCall
          ? { interruptAfter: ["tools"] }
          : { interruptBefore: ["tools"] }),
      });
      // Update thread list when continuing stream
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate, recursionLimit]
  );

  const markCurrentThreadAsResolved = useCallback(() => {
    stream.submit(null, {
      command: { goto: "__end__", update: null },
      config: { ...(activeAssistant?.config ?? {}), recursion_limit: recursionLimit },
    });
    // Update thread list when marking thread as resolved
    onHistoryRevalidate?.();
  }, [stream, onHistoryRevalidate, activeAssistant?.config, recursionLimit]);

  const resumeInterrupt = useCallback(
    (value: any) => {
      stream.submit(null, {
        command: { resume: value },
        config: { ...(activeAssistant?.config ?? {}), recursion_limit: recursionLimit },
      });
      // Update thread list when resuming from interrupt
      onHistoryRevalidate?.();
    },
    [stream, onHistoryRevalidate, activeAssistant?.config, recursionLimit]
  );

  const stopStream = useCallback(() => {
    stream.stop();
  }, [stream]);

  return {
    stream,
    todos: stream.values.todos ?? [],
    files: localFiles,
    email: stream.values.email,
    ui: stream.values.ui,
    setFiles,
    messages: stream.messages,
    isLoading: stream.isLoading,
    isThreadLoading: stream.isThreadLoading,
    interrupt: stream.interrupt,
    getMessagesMetadata: stream.getMessagesMetadata,
    sendMessage,
    runSingleStep,
    continueStream,
    stopStream,
    markCurrentThreadAsResolved,
    resumeInterrupt,
  };
}
