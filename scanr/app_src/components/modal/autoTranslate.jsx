import React from "react";
import { FiX, FiFolder } from "react-icons/fi";

import {
  locale,
  nativeAlert,
  exportActiveDocFlattenPng,
  createBubbleRectanglesGroup,
  renumberBubbleRectanglesGroup,
  createTranslatedTextLayers,
  applyCleanedPngToActiveDoc,
  openFolderPath,
  serverCreateJob,
  serverStartDetectBubbles,
  serverGetJob,
  serverGetBubblesAuto,
  serverSubmitBubbles,
  serverStartOcrClean,
  serverGetTranslations,
} from "../../utils";
import { useContext } from "../../context";
import { DEFAULT_BUBBLE_CLASS_CONFIDENCE, normalizeBubbleClassMap } from "../../bubbleClasses";

const GROUP_BUBBLES = "BUBBLES_DETECTED";
const GROUP_TRANSLATION = "TRANSLATION";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const extractBaseFontInfo = (style) => {
  const ts = style?.textProps?.layerText?.textStyleRange?.[0]?.textStyle || null;
  if (!ts || typeof ts !== "object") return null;

  const out = {};
  if (ts.fontPostScriptName) out.fontPostScriptName = ts.fontPostScriptName;
  if (ts.fontName) out.fontName = ts.fontName;
  if (ts.fontFamily) out.fontFamily = ts.fontFamily;
  if (ts.fontStyleName) out.fontStyleName = ts.fontStyleName;
  return Object.keys(out).length ? out : null;
};

const mergeBubbleClassOverrides = (bubbles, styles, bubbleClasses, classMap) => {
  const merged = { ...(styles || {}) };
  const bubbleMap = bubbleClasses || {};
  const mapping = normalizeBubbleClassMap(classMap || {});
  const fallbackConf = DEFAULT_BUBBLE_CLASS_CONFIDENCE;

  for (const b of bubbles || []) {
    const id = b?.id;
    if (!id) continue;
    const info = bubbleMap[id] || {};
    const label = info.class || info.label;
    if (!label) continue;
    const entry = mapping[label];
    if (!entry) continue;
    if (info.definite === false) continue;
    if (info.definite === undefined && info.confidence != null && info.confidence < fallbackConf) continue;

    const fontOverride = {};
    const hasFontFamily = !!(entry.fontPostScriptName || entry.fontName || entry.fontFamily);
    const hasFontStyle = !!entry.fontStyleName;
    if (hasFontFamily) {
      fontOverride.fontPostScriptName = entry.fontPostScriptName || entry.fontName || entry.fontFamily;
      fontOverride.fontName = entry.fontName || entry.fontFamily;
    }
    if (hasFontStyle) {
      fontOverride.fontStyleName = entry.fontStyleName;
    }
    if (entry.syntheticItalic === true || hasFontFamily || hasFontStyle) {
      fontOverride.syntheticItalic = !!entry.syntheticItalic;
    }
    if (Object.keys(fontOverride).length === 0) continue;

    merged[id] = { ...(merged[id] || {}), font: fontOverride };
  }

  return merged;
};

const AutoTranslateModal = React.memo(function AutoTranslateModal() {
  const context = useContext();
  const [busy, setBusy] = React.useState(false);
  const [job, setJob] = React.useState(null);
  const [bubblesFinal, setBubblesFinal] = React.useState([]);
  const [statusText, setStatusText] = React.useState("");

  const mountedRef = React.useRef(true);
  const busyRef = React.useRef(false);
  const cancelRef = React.useRef(false);
  const abortRef = React.useRef(null);

  React.useEffect(() => {
    return () => {
      mountedRef.current = false;
      cancelRef.current = true;
      try {
        abortRef.current && abortRef.current.abort && abortRef.current.abort();
      } catch (e) {}
      abortRef.current = null;
    };
  }, []);

  const safeSetBusy = (v) => {
    if (mountedRef.current) setBusy(v);
  };
  const safeSetJob = (j) => {
    if (mountedRef.current) setJob(j);
  };
  const safeSetBubblesFinal = (b) => {
    if (mountedRef.current) setBubblesFinal(b);
  };
  const safeSetStatusText = (t) => {
    if (mountedRef.current) setStatusText(t);
  };

  const isCancelledError = (e) => !!(e && (e.__cancelled || e.name === "AbortError"));
  const throwIfCancelled = () => {
    if (!mountedRef.current || cancelRef.current) {
      const err = new Error("cancelled");
      err.__cancelled = true;
      throw err;
    }
  };

  const cancelCurrentOperation = () => {
    cancelRef.current = true;
    try {
      abortRef.current && abortRef.current.abort && abortRef.current.abort();
    } catch (e) {}
    abortRef.current = null;
  };

  const close = () => {
    cancelCurrentOperation();
    context.dispatch({ type: "setModal" });
  };

  const pollJobUntil = async (
    jobId,
    predicate,
    { intervalMs = 1000, timeoutMs = 10 * 60 * 1000, stallTimeoutMs = 30 * 60 * 1000 } = {}
  ) => {
    const started = Date.now();
    let lastProgressSig = null;
    let lastProgressAt = Date.now();
    while (true) {
      throwIfCancelled();
      const j = await serverGetJob(jobId, { signal: abortRef.current ? abortRef.current.signal : undefined });
      if (!cancelRef.current) safeSetJob(j);
      if (j?.status === "error") {
        throw new Error(j?.last_error || "Job failed");
      }
      if (predicate(j)) return j;

      const progress = j?.progress || {};
      const sig = [j?.status || "", progress?.phase || "", progress?.done || 0, progress?.total || 0, progress?.pct || 0].join("|");
      if (sig !== lastProgressSig) {
        lastProgressSig = sig;
        lastProgressAt = Date.now();
      }

      const now = Date.now();
      if (timeoutMs != null && now - started > timeoutMs) {
        throw new Error("Timeout while waiting for server job.");
      }
      if (stallTimeoutMs != null && now - lastProgressAt > stallTimeoutMs) {
        throw new Error("Timeout while waiting for server job.");
      }
      await sleep(intervalMs);
    }
  };

  const detectBubbles = async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    cancelRef.current = false;
    abortRef.current = typeof AbortController !== "undefined" ? new AbortController() : null;
    const signal = abortRef.current ? abortRef.current.signal : undefined;

    safeSetBusy(true);
    safeSetStatusText(locale.autoTranslateStatusExport || "Exporting document...");
    try {
      throwIfCancelled();
      const exported = await exportActiveDocFlattenPng();

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusCreateJob || "Creating job...");
      const dpi = exported?.resolution != null ? Number(exported.resolution) : null;
      const createdJob = await serverCreateJob({
        sourcePngPath: exported.path,
        config: dpi ? { autofit: { dpi } } : undefined,
        signal,
      });
      safeSetJob(createdJob);

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusDetect || "Detecting bubbles...");
      await serverStartDetectBubbles(createdJob.job_id, { signal });

      const doneJob = await pollJobUntil(createdJob.job_id, (j) => j?.status === "await_edit");

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusFetchBubbles || "Fetching bubbles...");
      const bubblesAuto = await serverGetBubblesAuto(createdJob.job_id, { signal });
      const bubbles = bubblesAuto?.bubbles || [];

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusDrawRects || "Drawing rectangles in Photoshop...");
      await createBubbleRectanglesGroup({ groupName: GROUP_BUBBLES, bubbles, strokePx: 4, fillOpacity: 0 });

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusReadyEdit || "Ready: edit rectangles and click Finalize/Renumber.");
      safeSetJob(doneJob);

      // Cleanup temp export if possible (server already copied it to the job folder).
      try {
        window.cep.fs.deleteFile(exported.path);
      } catch (e) {}
    } catch (e) {
      if (!isCancelledError(e)) {
        nativeAlert((locale.autoTranslateError || "Auto Translate failed.") + "\n\n" + (e?.message || e), locale.errorTitle, true);
      }
    } finally {
      abortRef.current = null;
      busyRef.current = false;
      safeSetBusy(false);
    }
  };

  const finalizeAndSubmit = async () => {
    if (!job?.job_id) return;
    if (busyRef.current) return;
    busyRef.current = true;
    cancelRef.current = false;
    abortRef.current = typeof AbortController !== "undefined" ? new AbortController() : null;
    const signal = abortRef.current ? abortRef.current.signal : undefined;

    safeSetBusy(true);
    safeSetStatusText(locale.autoTranslateStatusRenumber || "Renumbering rectangles...");
    try {
      throwIfCancelled();
      const readingOrder = context.state.bubbleReadingOrder || "ltr";
      const bubbles = await renumberBubbleRectanglesGroup(GROUP_BUBBLES, 0.5, readingOrder);
      safeSetBubblesFinal(bubbles);

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusSubmit || "Submitting bubbles to server...");
      await serverSubmitBubbles(job.job_id, bubbles, { signal });

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusReadyOcr || "Ready: run OCR + Clean.");
    } catch (e) {
      if (!isCancelledError(e)) {
        nativeAlert((locale.autoTranslateError || "Auto Translate failed.") + "\n\n" + (e?.message || e), locale.errorTitle, true);
      }
    } finally {
      abortRef.current = null;
      busyRef.current = false;
      safeSetBusy(false);
    }
  };

  const runOcrClean = async () => {
    if (!job?.job_id) return;
    if (busyRef.current) return;
    busyRef.current = true;
    cancelRef.current = false;
    abortRef.current = typeof AbortController !== "undefined" ? new AbortController() : null;
    const signal = abortRef.current ? abortRef.current.signal : undefined;

    safeSetBusy(true);
    safeSetStatusText(locale.autoTranslateStatusOcrClean || "Running OCR + Clean...");
    try {
      throwIfCancelled();
      await serverStartOcrClean(job.job_id, { signal });
      const doneJob = await pollJobUntil(job.job_id, (j) => j?.status === "await_translation", {
        timeoutMs: null,
        stallTimeoutMs: null,
      });

      throwIfCancelled();
      try {
        safeSetStatusText(locale.autoTranslateStatusApplyCleaned || "Applying cleaned image into Photoshop...");
        const cleanedPath = doneJob?.paths_open?.cleaned_png || doneJob?.paths?.cleaned_png;
        if (cleanedPath) {
          await applyCleanedPngToActiveDoc({ path: cleanedPath, belowGroupName: GROUP_BUBBLES, translationGroupName: GROUP_TRANSLATION });
        }
      } catch (eApply) {
        // Don't fail the whole flow if PSD update fails; user can still open cleaned.png from job folder.
        console.error("applyCleanedPng failed", eApply);
      }

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusAwaitTranslation || "Waiting for translate.txt (put it in the job folder).");
      safeSetJob(doneJob);
    } catch (e) {
      if (!isCancelledError(e)) {
        nativeAlert((locale.autoTranslateError || "Auto Translate failed.") + "\n\n" + (e?.message || e), locale.errorTitle, true);
      }
    } finally {
      abortRef.current = null;
      busyRef.current = false;
      safeSetBusy(false);
    }
  };

  const openJobFolder = () => {
    const jobDir = job?.paths_open?.job_dir || job?.paths?.job_dir;
    if (jobDir) openFolderPath(jobDir);
  };

  const applyTranslation = async () => {
    if (!job?.job_id) return;
    if (busyRef.current) return;
    busyRef.current = true;
    cancelRef.current = false;
    abortRef.current = typeof AbortController !== "undefined" ? new AbortController() : null;
    const signal = abortRef.current ? abortRef.current.signal : undefined;

    safeSetBusy(true);
    safeSetStatusText(locale.autoTranslateStatusApply || "Applying translation...");
    try {
      throwIfCancelled();
      const style = context.state.currentStyle || null;
      const direction = context.state.direction;

      const baseFont = extractBaseFontInfo(style);
      const normalizedClassMap = normalizeBubbleClassMap(context.state.bubbleClassMap || {});
      const data = await serverGetTranslations(job.job_id, {
        autofit: {
          font_base: baseFont,
          bubble_class_map: normalizedClassMap,
          bubble_class_conf: DEFAULT_BUBBLE_CLASS_CONFIDENCE,
        },
      }, { signal });
      const translations = data?.translations || [];
      const styles = data?.styles || null;
      const fitMap = data?.fit || null;
      const bubbleClasses = data?.bubble_classes || {};
      const mergedStyles = mergeBubbleClassOverrides(bubblesFinal, styles, bubbleClasses, context.state.bubbleClassMap);

      throwIfCancelled();
      await createTranslatedTextLayers({
        groupName: GROUP_TRANSLATION,
        bubbles: bubblesFinal,
        translations,
        style,
        // Spec: textbox should match bubble bbox (no internal padding).
        fit: { minSize: 16, maxSize: 34, padding: 0, map: fitMap },
        styles: mergedStyles,
        direction,
      });

      throwIfCancelled();
      safeSetStatusText(locale.autoTranslateStatusDone || "Done.");
    } catch (e) {
      if (!isCancelledError(e)) {
        nativeAlert((locale.autoTranslateError || "Auto Translate failed.") + "\n\n" + (e?.message || e), locale.errorTitle, true);
      }
    } finally {
      abortRef.current = null;
      busyRef.current = false;
      safeSetBusy(false);
    }
  };

  const jobProgress = job?.progress?.phase ? `${job.progress.phase} (${job.progress.pct || 0}%)` : "";

  return (
    <React.Fragment>
      <div className="app-modal-header hostBgdDark">
        <div className="app-modal-title">{locale.autoTranslateTitle || "Auto Translate"}</div>
        <button className="topcoat-icon-button--large--quiet" onClick={close} title={locale.close || "Close"}>
          <FiX />
        </button>
      </div>

      <div className="app-modal-body">
        <div className="app-modal-body-inner">
          <div className="field">
            <button className="topcoat-button--large--cta" onClick={detectBubbles} disabled={busy}>
              {locale.autoTranslateDetectBubbles || "1) Detect bubbles (auto)"}
            </button>
            <div className="field-descr">{locale.autoTranslateDetectBubblesDescr || `Creates ${GROUP_BUBBLES} group with blue 4px rectangles.`}</div>
          </div>

          <div className="field hostBrdTopContrast">
            <button className="topcoat-button--large--cta" onClick={finalizeAndSubmit} disabled={busy || !job?.job_id}>
              {locale.autoTranslateFinalize || "2) Finalize / Renumber"}
            </button>
            <div className="field-descr">{locale.autoTranslateFinalizeDescr || "Reads rectangles, sorts them in reading order and renames to B001.."} </div>
          </div>

          <div className="field hostBrdTopContrast">
            <button className="topcoat-button--large--cta" onClick={runOcrClean} disabled={busy || !job?.job_id}>
              {locale.autoTranslateOcrClean || "3) Extract original + Clean"}
            </button>
            <div className="field-descr">{locale.autoTranslateOcrCleanDescr || "Generates original.txt, cleaned.png and frames/ (if enabled)."} </div>
          </div>

          <div className="field hostBrdTopContrast">
            <button className="topcoat-button--large--cta" onClick={applyTranslation} disabled={busy || !job?.job_id || !bubblesFinal.length}>
              {locale.autoTranslateApply || "4) Apply translation"}
            </button>
            <div className="field-descr">{locale.autoTranslateApplyDescr || `Reads translate.txt from the job folder and creates ${GROUP_TRANSLATION} text layers.`}</div>
          </div>

          <div className="field hostBrdTopContrast">
            <button className="topcoat-button--large--quiet m-cta" onClick={openJobFolder} disabled={!job?.job_id}>
              <FiFolder />
              {locale.autoTranslateOpenJobFolder || "Open job folder"}
            </button>
            <div className="field-descr">
              {(job?.paths_open?.job_dir || job?.paths?.job_dir || "").toString()}
            </div>
          </div>
        </div>
      </div>

      <div className="app-modal-footer hostBgdDark">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          <div style={{ opacity: 0.9 }}>{statusText}</div>
          <div style={{ opacity: 0.7 }}>{jobProgress}</div>
        </div>
      </div>
    </React.Fragment>
  );
});

export default AutoTranslateModal;
