const DEFAULT_BUBBLE_CLASS_CONFIDENCE = 0.6;
const PSEUDO_ITALIC_VALUE = "__pseudo_italic__";

const BUBBLE_CLASSES = [
  { id: "elipse", labelKey: "bubbleClassElipse", labelFallback: "Ellipse" },
  { id: "cloude", labelKey: "bubbleClassCloude", labelFallback: "Cloud" },
  { id: "other", labelKey: "bubbleClassOther", labelFallback: "Other" },
  { id: "rectangle", labelKey: "bubbleClassRectangle", labelFallback: "Rectangle" },
  { id: "sea_uchirin", labelKey: "bubbleClassSeaUchirin", labelFallback: "Sea uchirin" },
  { id: "thorn", labelKey: "bubbleClassThorn", labelFallback: "Thorn" },
];

const DEFAULT_BUBBLE_CLASS_MAP = {
  elipse: {
    fontFamily: "",
    fontStyleName: "",
    fontPostScriptName: "",
    fontName: "",
    syntheticItalic: false,
  },
  cloude: {
    fontFamily: "Teslics Document Cyr",
    fontStyleName: "Regular",
    fontPostScriptName: "Teslics Document Cyr",
    fontName: "Teslics Document Cyr",
    syntheticItalic: false,
  },
  other: {
    fontFamily: "",
    fontStyleName: "",
    fontPostScriptName: "",
    fontName: "",
    syntheticItalic: false,
  },
  rectangle: {
    fontFamily: "v_CCMonologous",
    fontStyleName: "Regular",
    fontPostScriptName: "v_CCMonologous",
    fontName: "v_CCMonologous",
    syntheticItalic: false,
  },
  sea_uchirin: {
    fontFamily: "v_CCWild Words Roman",
    fontStyleName: "Regular",
    fontPostScriptName: "v_CCWildWordsRoman",
    fontName: "v_CCWild Words Roman",
    syntheticItalic: true,
  },
  thorn: {
    fontFamily: "CCWildWords",
    fontStyleName: "Italic",
    fontPostScriptName: "CCWildWords Italic",
    fontName: "CCWildWords Italic",
    syntheticItalic: false,
  },
};

const normalizeBubbleClassMap = (raw = {}) => {
  const base = JSON.parse(JSON.stringify(DEFAULT_BUBBLE_CLASS_MAP));
  if (raw && typeof raw === "object") {
    Object.keys(base).forEach((key) => {
      if (raw[key] && typeof raw[key] === "object") {
        const { useDefault, ...rest } = raw[key];
        base[key] = { ...base[key], ...rest };
      }
    });
  }
  return base;
};

export {
  BUBBLE_CLASSES,
  DEFAULT_BUBBLE_CLASS_CONFIDENCE,
  DEFAULT_BUBBLE_CLASS_MAP,
  PSEUDO_ITALIC_VALUE,
  normalizeBubbleClassMap,
};
