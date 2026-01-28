import _ from "lodash";
import React from "react";

import { csInterface, setActiveLayerText, createTextLayerInSelection, createTextLayersInStoredSelections, alignTextLayerToSelection, alignTextLayerToBubbleAI, fixTrailingSpacesInActiveLayer, getHotkeyPressed, changeActiveLayerTextSize } from "./utils";
import { useContext } from "./context";

const CTRL = "CTRL";
const SHIFT = "SHIFT";
const ALT = "ALT";
const WIN = "WIN";

const intervalTime = 120;
let keyUp = true;
let lastAction = 0;

const checkRepeatTime = (time = 0) => {
  const now = Date.now();
  if (!keyUp || now - lastAction < time) return false;
  lastAction = now;
  keyUp = false;
  return true;
};

const checkShortcut = (state, ref) => {
  return ref.every((key) => state.includes(key));
};

const HotkeysListner = React.memo(function HotkeysListner() {
  const context = useContext();
  const stateRef = React.useRef(context.state);

  React.useEffect(() => {
    stateRef.current = context.state;
  }, [context.state]);

  const checkState = React.useCallback((state) => {
    const current = stateRef.current;
    const realState = (state || "").split("a");
    realState.shift();
    realState.pop();
    if (checkShortcut(realState, current.shortcut.add)) {
      if (!checkRepeatTime()) return;

      const storedSelections = current.storedSelections || [];

      if (current.multiBubbleMode && storedSelections.length > 0) {
        // Mode sélections multiples
        const texts = [];
        const styles = [];
        const lines = current.lines || [];
        let nextFallbackIndex = current.currentLineIndex;

        const resolveStyleForLine = (targetLine, selection) => {
          if (targetLine?.style) {
            return targetLine.style;
          }
          if (selection?.styleId) {
            const storedStyle = current.styles.find((s) => s.id === selection.styleId);
            if (storedStyle) return storedStyle;
          }
          return current.currentStyle;
        };

        const resolveLineForSelection = (selection) => {
          if (typeof selection.lineIndex === "number" && selection.lineIndex >= 0) {
            const storedLine = lines[selection.lineIndex];
            if (storedLine && !storedLine.ignore) {
              nextFallbackIndex = Math.max(nextFallbackIndex, selection.lineIndex + 1);
              return storedLine;
            }
          }

          while (nextFallbackIndex < lines.length) {
            const candidate = lines[nextFallbackIndex];
            nextFallbackIndex++;
            if (candidate && !candidate.ignore) {
              return candidate;
            }
          }
          return null;
        };

        for (let i = 0; i < storedSelections.length; i++) {
          const selection = storedSelections[i];
          const targetLine = resolveLineForSelection(selection);
          if (!targetLine) {
            break;
          }

          texts.push(targetLine.text);

          let lineStyle = resolveStyleForLine(targetLine, selection);
          if (lineStyle && current.textScale) {
            lineStyle = _.cloneDeep(lineStyle);
            const txtStyle = lineStyle.textProps?.layerText.textStyleRange?.[0]?.textStyle || {};
            if (typeof txtStyle.size === "number") {
              txtStyle.size *= current.textScale / 100;
            }
            if (typeof txtStyle.leading === "number" && txtStyle.leading) {
              txtStyle.leading *= current.textScale / 100;
            }
          }
          styles.push(lineStyle);
        }

        const pointText = current.pastePointText;
        const padding = current.internalPadding || 0;
        createTextLayersInStoredSelections(texts, styles, storedSelections, pointText, padding, (ok) => {
          if (ok) {
            // Vider les sélections stockées
            context.dispatch({ type: "clearSelections" });
          }
        });
      } else {
        // Mode sélection unique (comportement original)
        const line = current.currentLine || { text: "" };
        let style = current.currentStyle;
        if (style && current.textScale) {
          style = _.cloneDeep(style);
          const txtStyle = style.textProps?.layerText.textStyleRange?.[0]?.textStyle || {};
          if (typeof txtStyle.size === "number") {
            txtStyle.size *= current.textScale / 100;
          }
          if (typeof txtStyle.leading === "number" && txtStyle.leading) {
            txtStyle.leading *= current.textScale / 100;
          }
        }
        const pointText = current.pastePointText;
        const padding = current.internalPadding || 0;
        createTextLayerInSelection(line.text, style, pointText, padding, (ok) => {
          if (ok) context.dispatch({ type: "nextLine", add: true });
        });
      }
    } else if (checkShortcut(realState, current.shortcut.apply)) {
      if (!checkRepeatTime()) return;
      const line = current.currentLine || { text: "" };
      let style = current.currentStyle;
      if (style && current.textScale) {
        style = _.cloneDeep(style);
        const txtStyle = style.textProps?.layerText.textStyleRange?.[0]?.textStyle || {};
        if (typeof txtStyle.size === "number") {
          txtStyle.size *= current.textScale / 100;
        }
        if (typeof txtStyle.leading === "number" && txtStyle.leading) {
          txtStyle.leading *= current.textScale / 100;
        }
      }
      setActiveLayerText(line.text, style, current.direction, (ok) => {
        if (ok) context.dispatch({ type: "nextLine", add: true });
      });
    } else if (checkShortcut(realState, current.shortcut.center)) {
      if (!checkRepeatTime()) return;
      const padding = current.internalPadding || 0;
      // Fix trailing spaces first, then align
      fixTrailingSpacesInActiveLayer().then(() => {
        if (current.useAI) {
          alignTextLayerToBubbleAI(current.resizeTextBoxOnCenter, padding).catch((e) => console.error(e));
        } else {
          alignTextLayerToSelection(current.resizeTextBoxOnCenter, padding);
        }
      });
    } else if (checkShortcut(realState, current.shortcut.toggleMultiBubble)) {
      if (!checkRepeatTime(300)) return;
      context.dispatch({ type: "setMultiBubbleMode", value: !current.multiBubbleMode });
    } else if (checkShortcut(realState, current.shortcut.next)) {
      if (!checkRepeatTime(300)) return;
      context.dispatch({ type: "nextLine" });
    } else if (checkShortcut(realState, current.shortcut.previous)) {
      if (!checkRepeatTime(300)) return;
      context.dispatch({ type: "prevLine" });
    } else if (checkShortcut(realState, current.shortcut.increase)) {
      if (!checkRepeatTime(300)) return;
      changeActiveLayerTextSize(current.textSizeIncrement || 1);
    } else if (checkShortcut(realState, current.shortcut.decrease)) {
      if (!checkRepeatTime(300)) return;
      changeActiveLayerTextSize(-(current.textSizeIncrement || 1));
    } else if (checkShortcut(realState, current.shortcut.insertText)) {
      if (!checkRepeatTime()) return;
      const line = current.currentLine || { text: "" };
      setActiveLayerText(line.text, null, current.direction, (ok) => {
        if (ok) context.dispatch({ type: "nextLine", add: true });
      });
    } else if (checkShortcut(realState, current.shortcut.nextPage)) {
      if (!checkRepeatTime(300)) return;
      context.dispatch({ type: "nextPage" });
    } else {
      keyUp = true;
    }

  }, [context.dispatch]);

  React.useEffect(() => {
    let cancelled = false;
    let timeoutId = null;

    const tick = () => {
      if (cancelled) return;
      const current = stateRef.current;

      // Reduce host calls when panel is hidden or settings modal is open
      if (document.hidden || current.modalType === "settings") {
        timeoutId = setTimeout(tick, 250);
        return;
      }

      getHotkeyPressed((raw) => {
        if (cancelled) return;
        try {
          checkState(raw || "");
        } catch (e) {
          // Avoid breaking the polling loop due to runtime errors.
          // eslint-disable-next-line no-console
          console.error(e);
        } finally {
          timeoutId = setTimeout(tick, intervalTime);
        }
      });
    };

    tick();

    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [checkState]);

  React.useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        if (stateRef.current.modalType) {
          context.dispatch({ type: "setModal" });
        }
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [context.dispatch]);

  React.useEffect(() => {
    const keyInterests = [{ keyCode: 27 }];
    csInterface.registerKeyEventsInterest(JSON.stringify(keyInterests));
  }, []);

  return <React.Fragment />;
});

export default HotkeysListner;
