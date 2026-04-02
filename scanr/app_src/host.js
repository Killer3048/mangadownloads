/* globals app, documents, activeDocument, ScriptUI, DialogModes, LayerKind, ActionReference, ActionDescriptor, executeAction, executeActionGet, stringIDToTypeID, jamEngine, jamJSON, jamText */

var charID = {
  Back: 1113678699, // 'Back'
  Background: 1113811815, // 'Bckg'
  Bottom: 1114926957, // 'Btom'
  By: 1115234336, // 'By  '
  Channel: 1130917484, // 'Chnl'
  Contract: 1131312227, // 'Cntc'
  Document: 1147366766, // 'Dcmn'
  Expand: 1165521006, // 'Expn'
  FrameSelect: 1718838636, // 'fsel'
  Horizontal: 1215461998, // 'Hrzn'
  Layer: 1283027488, // 'Lyr '
  Left: 1281713780, // 'Left'
  Move: 1836021349, // 'move'
  None: 1315925605, // 'None'
  Null: 1853189228, // 'null'
  Offset: 1332114292, // 'Ofst'
  Ordinal: 1332896878, // 'Ordn'
  PixelUnit: 592476268, // '#Pxl'
  Point: 1349415968, // 'Pnt '
  Property: 1349677170, // 'Prpr'
  Right: 1382508660, // 'Rght'
  Select: 1936483188, // 'slct'
  Set: 1936028772, // 'setd'
  Size: 1400512544, // 'Sz  '
  Target: 1416783732, // 'Trgt'
  Text: 1417180192, // 'Txt '
  TextLayer: 1417170034, // 'TxLr'
  TextShapeType: 1413830740, // 'TEXT'
  TextStyle: 1417180243, // 'TxtS'
  TextStyleRange: 1417180276, // 'Txtt'
  To: 1411391520, // 'T   '
  Top: 1416589344, // 'Top '
  Vertical: 1450341475, // 'Vrtc'
};

var _SAFE_PARAGRAPH_PROPS = [
  "align",
  "alignment",
  "firstLineIndent",
  "startIndent",
  "endIndent",
  "spaceBefore",
  "spaceAfter",
  "autoLeadingPercentage",
  "leadingType",
  "hyphenate",
  "hyphenateWordSize",
  "hyphenatePreLength",
  "hyphenatePostLength",
  "hyphenateLimit",
  "hyphenationZone",
  "hyphenateCapitalized",
  "hangingRoman",
  "burasagari",
  "textEveryLineComposer",
  "textComposerEngine",
];

var _DEFAULT_SELECTION_SCALE = 0.9;
var _MIN_TEXTBOX_WIDTH = 10;
var _MIN_TEXTBOX_HEIGHT = 10;
var _TEMP_SELECTION_CHANNEL = "__TyperSelectionTemp__";
var _DEFAULT_ADJUST_SEQUENCE = [-5, -5, -5, -5, -5, -5, 5, 5, 5, 5, 5, 5];

var _hostState = {
  fallbackTextSize: 20,
  applyCleanedPng: {
    data: null,
    result: "",
  },
  applyImaginePatches: {
    data: null,
    result: "",
  },
  setActiveLayerText: {
    data: null,
    result: "",
  },
  createTextLayerInSelection: {
    data: null,
    result: "",
    point: false,
    padding: 0,
  },
  alignTextLayerToSelection: {
    result: "",
    resize: false,
    padding: 0,
  },
  alignTextLayerToTarget: {
    result: "",
    resize: false,
    padding: 0,
    target: null,
  },
  changeActiveLayerTextSize: {
    value: 0,
    result: "",
  },
  selectionMonitor: {
    lastBoundsKey: null,
    callback: null,
    pendingSelection: null,
    pendingBoundsKey: null,
    lastPollMs: 0,
  },
  createTextLayersInStoredSelections: {
    data: null,
    result: "",
    point: false,
    padding: 0,
    selections: [],
  },
  createBubbleRectangles: {
    data: null,
    result: "",
  },
  renumberBubbleGroup: {
    data: null,
    result: "",
  },
  createTranslatedTextLayers: {
    data: null,
    result: "",
  },
  lastOpenedDocId: null,
};

function _clone(obj) {
  if (!obj || typeof obj !== "object") return obj;
  if (obj instanceof Array) {
    var arr = [];
    for (var i = 0; i < obj.length; i++) {
      arr[i] = _clone(obj[i]);
    }
    return arr;
  }
  var result = {};
  for (var key in obj) {
    if (obj.hasOwnProperty(key)) {
      result[key] = _clone(obj[key]);
    }
  }
  return result;
}

function _getHostDefaultStyle() {
  return {
    layerText: {
      textGridding: "none",
      orientation: "horizontal",
      antiAlias: "antiAliasSmooth",
      textStyleRange: [
        {
          from: 0,
          to: 100,
          textStyle: {
            fontPostScriptName: "v_CCWildWordsRoman",
            fontName: "v_CCWild Words Roman",
            fontStyleName: "Regular",
            fontScript: 0,
            fontTechnology: 1,
            fontAvailable: true,
            size: 14,
            impliedFontSize: 14,
            horizontalScale: 100,
            verticalScale: 100,
            autoLeading: true,
            tracking: 0,
            baselineShift: 0,
            impliedBaselineShift: 0,
            autoKern: "metricsKern",
            fontCaps: "normal",
            digitSet: "defaultDigits",
            diacXOffset: 0,
            markYDistFromBaseline: 100,
            otbaseline: "normal",
            ligature: false,
            altligature: false,
            connectionForms: false,
            contextualLigatures: false,
            baselineDirection: "withStream",
            color: { red: 0, green: 0, blue: 0 }
          }
        }
      ],
      paragraphStyleRange: [
        {
          from: 0,
          to: 100,
          paragraphStyle: {
            burasagari: "burasagariNone",
            singleWordJustification: "justifyAll",
            justificationMethodType: "justifMethodAutomatic",
            textEveryLineComposer: false,
            alignment: "center",
            hangingRoman: true,
            hyphenate: true
          }
        }
      ]
    },
    typeUnit: "pixelsUnit"
  };
}

function _getHostDefaultStroke() {
  return {
    enabled: false,
    size: 0,
    opacity: 100,
    position: "outer",
    color: { r: 255, g: 255, b: 255 }
  };
}

function _findInstalledFont(desiredPostScriptName, desiredFamily) {
  var ps = (desiredPostScriptName || "").toString().toLowerCase();
  var fam = (desiredFamily || "").toString().toLowerCase();
  var best = null;
  for (var i = 0; i < app.fonts.length; i++) {
    var f = app.fonts[i];
    if (!f) continue;
    var fPs = (f.postScriptName || f.name || "").toString().toLowerCase();
    var fFam = (f.family || "").toString().toLowerCase();
    if (ps && fPs === ps) {
      return f;
    }
    if (fam && (fFam === fam || fPs === fam)) {
      best = f;
    }
  }
  return best;
}

function _resolveDefaultFont() {
  var candidates = [
    { ps: "v_CCWildWordsRoman", family: "v_CCWild Words Roman" },
    { ps: "v_CCWild Words Roman", family: "v_CCWild Words Roman" },
    { ps: "CCWildWordsRoman", family: "CCWild Words Roman" },
    { ps: "CCWild Words Roman", family: "CCWild Words Roman" },
    { ps: "WildWordsRoman", family: "Wild Words Roman" },
  ];

  for (var i = 0; i < candidates.length; i++) {
    var c = candidates[i];
    var f = _findInstalledFont(c.ps, c.family);
    if (f) {
      return {
        fontPostScriptName: f.postScriptName || f.name || c.ps,
        fontName: f.family || c.family,
        fontStyleName: f.style || "Regular",
      };
    }
  }

  return { fontPostScriptName: "Tahoma", fontName: "Tahoma", fontStyleName: "Regular" };
}

function _ensureStyle(style) {
  var normalized = style ? _clone(style) : {};
  if (!normalized.textProps || !normalized.textProps.layerText) {
    normalized.textProps = _getHostDefaultStyle();
  }
  if (typeof normalized.stroke === "undefined") {
    normalized.stroke = _getHostDefaultStroke();
  }

  // Ensure font names match an installed font (otherwise Photoshop silently falls back).
  try {
    var ts = normalized.textProps.layerText.textStyleRange[0].textStyle;
    var desiredPs = ts.fontPostScriptName || ts.fontName || "";
    var desiredFamily = ts.fontName || "";
    var fnt = _findInstalledFont(desiredPs, desiredFamily);
    if (!fnt) {
      var df = _resolveDefaultFont();
      ts.fontPostScriptName = df.fontPostScriptName;
      ts.fontName = df.fontName;
      ts.fontStyleName = df.fontStyleName;
      ts.fontAvailable = true;
    } else {
      ts.fontPostScriptName = fnt.postScriptName || fnt.name || ts.fontPostScriptName;
      ts.fontName = fnt.family || ts.fontName;
      ts.fontStyleName = fnt.style || ts.fontStyleName;
      ts.fontAvailable = true;
    }
    normalized.textProps.layerText.textStyleRange[0].textStyle = ts;
  } catch (eFont) { }

  return normalized;
}

function _normalizeRgb(rgb) {
  if (!rgb) return null;
  var r = rgb.r != null ? rgb.r : rgb.red;
  var g = rgb.g != null ? rgb.g : rgb.green;
  var b = rgb.b != null ? rgb.b : rgb.blue;
  if (r == null || g == null || b == null) return null;
  return {
    r: Math.max(0, Math.min(255, Math.round(r))),
    g: Math.max(0, Math.min(255, Math.round(g))),
    b: Math.max(0, Math.min(255, Math.round(b))),
  };
}

function _applyStyleOverrides(baseStyle, overrides) {
  var style = _clone(baseStyle || {});
  if (!overrides) return style;

  var fill = _normalizeRgb(overrides.fill);
  if (fill && style.textProps && style.textProps.layerText && style.textProps.layerText.textStyleRange) {
    var ts = style.textProps.layerText.textStyleRange[0].textStyle || {};
    ts.color = { red: fill.r, green: fill.g, blue: fill.b };
    style.textProps.layerText.textStyleRange[0].textStyle = ts;
  }

  var stroke = overrides.stroke || null;
  if (stroke) {
    var nextStroke = _clone(style.stroke || _getHostDefaultStroke());
    if (typeof stroke.enabled !== "undefined") nextStroke.enabled = !!stroke.enabled;
    if (stroke.size != null) nextStroke.size = stroke.size;
    if (stroke.opacity != null) nextStroke.opacity = stroke.opacity;
    if (stroke.position) nextStroke.position = stroke.position;
    var strokeColor = _normalizeRgb(stroke.color || stroke);
    if (strokeColor) nextStroke.color = strokeColor;
    style.stroke = nextStroke;
  }

  var font = overrides.font || overrides;
  if (font && style.textProps && style.textProps.layerText && style.textProps.layerText.textStyleRange) {
    var ts2 = style.textProps.layerText.textStyleRange[0].textStyle || {};
    if (font.fontPostScriptName) ts2.fontPostScriptName = font.fontPostScriptName;
    if (font.fontName) ts2.fontName = font.fontName;
    if (font.fontStyleName) ts2.fontStyleName = font.fontStyleName;
    if (font.syntheticItalic !== undefined) ts2.syntheticItalic = !!font.syntheticItalic;
    if (font.syntheticBold !== undefined) ts2.syntheticBold = !!font.syntheticBold;
    style.textProps.layerText.textStyleRange[0].textStyle = ts2;
  }

  return _ensureStyle(style);
}

function _applyAutoTranslateDefaults(style) {
  if (!style || !style.textProps || !style.textProps.layerText) return style;
  var layerText = style.textProps.layerText;

  // Auto-translate defaults: optical kerning, crisp/sharp AA, TT off.
  layerText.antiAlias = "antiAliasCrisp";

  if (layerText.textStyleRange && layerText.textStyleRange.length) {
    var ts = layerText.textStyleRange[0].textStyle || {};
    ts.autoKern = "opticalKern";
    ts.fontCaps = ts.fontCaps || "normal";
    layerText.textStyleRange[0].textStyle = ts;
  }

  if (layerText.paragraphStyleRange && layerText.paragraphStyleRange.length) {
    var ps = layerText.paragraphStyleRange[0].paragraphStyle || {};
    ps.textEveryLineComposer = false;
    layerText.paragraphStyleRange[0].paragraphStyle = ps;
  }

  style.textProps.layerText = layerText;
  return style;
}

function _changeToPointText() {
  try {
    if (app.activeDocument && app.activeDocument.activeLayer && app.activeDocument.activeLayer.textItem) {
      app.activeDocument.activeLayer.textItem.kind = TextType.POINTTEXT;
      return;
    }
  } catch (e) { }
  var reference = new ActionReference();
  reference.putProperty(charID.Property, charID.TextShapeType);
  reference.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);
  var descriptor = new ActionDescriptor();
  descriptor.putReference(charID.Null, reference);
  descriptor.putEnumerated(charID.To, charID.TextShapeType, charID.Point);
  executeAction(charID.Set, descriptor, DialogModes.NO);
}

function _changeToBoxText() {
  var reference = new ActionReference();
  reference.putProperty(charID.Property, charID.TextShapeType);
  reference.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);
  var descriptor = new ActionDescriptor();
  descriptor.putReference(charID.Null, reference);
  descriptor.putEnumerated(charID.To, charID.TextShapeType, stringIDToTypeID("box"));
  executeAction(charID.Set, descriptor, DialogModes.NO);
}

function _layerIsTextLayer() {
  var layer = _getCurrent(charID.Layer, charID.Text);
  return layer.hasKey(charID.Text);
}

function _textLayerIsPointText() {
  var textKey = _getCurrent(charID.Layer, charID.Text).getObjectValue(charID.Text);
  var textType = textKey.getList(stringIDToTypeID("textShape")).getObjectValue(0).getEnumerationValue(charID.TextShapeType);
  return textType === charID.Point;
}

function _getTextLayerSize() {
  try {
    var textParams = jamText.getLayerText();
    if (textParams && textParams.layerText &&
      textParams.layerText.textStyleRange &&
      textParams.layerText.textStyleRange[0] &&
      textParams.layerText.textStyleRange[0].textStyle &&
      textParams.layerText.textStyleRange[0].textStyle.size) {
      return textParams.layerText.textStyleRange[0].textStyle.size;
    }
  } catch (e) { }
  return _hostState.fallbackTextSize || 20;
}

function _convertPixelToPoint(value) {
  return (parseInt(value) / activeDocument.resolution) * 72;
}

function _createCurrent(target, id) {
  var reference = new ActionReference();
  if (id > 0) reference.putProperty(charID.Property, id);
  reference.putEnumerated(target, charID.Ordinal, charID.Target);
  return reference;
}

function _getCurrent(target, id) {
  return executeActionGet(_createCurrent(target, id));
}

function _deselect() {
  var reference = new ActionReference();
  reference.putProperty(charID.Channel, charID.FrameSelect);
  var descriptor = new ActionDescriptor();
  descriptor.putReference(charID.Null, reference);
  descriptor.putEnumerated(charID.To, charID.Ordinal, charID.None);
  executeAction(charID.Set, descriptor, DialogModes.NO);
}

function _getDescNumber(desc, key) {
  try {
    return desc.getUnitDoubleValue(key);
  } catch (e1) { }
  try {
    return desc.getDouble(key);
  } catch (e2) { }
  try {
    return desc.getInteger(key);
  } catch (e3) { }
  return 0;
}

function _getBoundsFromDescriptor(bounds) {
  var top = _getDescNumber(bounds, charID.Top);
  var left = _getDescNumber(bounds, charID.Left);
  var right = _getDescNumber(bounds, charID.Right);
  var bottom = _getDescNumber(bounds, charID.Bottom);
  return {
    top: top,
    left: left,
    right: right,
    bottom: bottom,
    width: right - left,
    height: bottom - top,
    xMid: (left + right) / 2,
    yMid: (top + bottom) / 2,
  };
}

function _getCurrentSelectionBounds() {
  if (!documents.length) return;
  try {
    var doc = _getCurrent(charID.Document, charID.FrameSelect);
    if (doc.hasKey(charID.FrameSelect)) {
      var bounds = doc.getObjectValue(charID.FrameSelect);
      return _getBoundsFromDescriptor(bounds);
    }
  } catch (e) {
    // ignore
  }
}

function _getCurrentTextLayerBounds() {
  var boundsTypeId = stringIDToTypeID("bounds");
  var bounds = _getCurrent(charID.Layer, boundsTypeId).getObjectValue(boundsTypeId);
  return _getBoundsFromDescriptor(bounds);
}


function _getTopLevelLayerIndex(doc, layer) {
  if (!doc || !layer) return -1;
  try {
    var layers = doc.layers || [];
    for (var i = 0; i < layers.length; i++) {
      if (layers[i] && layers[i].id === layer.id) return i;
    }
  } catch (e) { }
  return -1;
}

function _moveLayerBelow(doc, layerToMove, referenceLayer) {
  if (!doc || !layerToMove || !referenceLayer) return;
  var tryMove = function (placement) {
    try {
      layerToMove.move(referenceLayer, placement);
      return true;
    } catch (e) {
      return false;
    }
  };

  // Prefer "below" but verify by index in the document layer stack.
  tryMove(ElementPlacement.PLACEAFTER);
  var idxMove = _getTopLevelLayerIndex(doc, layerToMove);
  var idxRef = _getTopLevelLayerIndex(doc, referenceLayer);
  if (idxMove >= 0 && idxRef >= 0 && idxMove < idxRef) {
    tryMove(ElementPlacement.PLACEBEFORE);
  }
}

function _modifySelectionBounds(amount) {
  if (amount == 0) return;
  var size = new ActionDescriptor();
  size.putUnitDouble(charID.By, charID.PixelUnit, Math.abs(amount));
  executeAction(amount > 0 ? charID.Expand : charID.Contract, size, DialogModes.NO);
}


function _getAdjustedSelectionBounds(bounds, amount) {
  if (!bounds || amount === 0) return bounds;

  var doc;
  try {
    doc = app.activeDocument;
  } catch (error) {
    doc = null;
  }

  if (!doc || !doc.selection) {
    return _getAdjustedSelectionBoundsFallback(bounds, amount);
  }

  var tempChannel = _createTempSelectionChannel(doc);
  if (!tempChannel) {
    return _getAdjustedSelectionBoundsFallback(bounds, amount);
  }

  var adjusted = null;
  try {
    _modifySelectionBounds(amount);
    adjusted = _getCurrentSelectionBounds();
  } catch (error2) {
    adjusted = null;
  } finally {
    try {
      doc.selection.load(tempChannel);
    } catch (restoreError) { }
    try {
      tempChannel.remove();
    } catch (removeError) { }
  }

  if (!adjusted) {
    return _getAdjustedSelectionBoundsFallback(bounds, amount);
  }
  return adjusted;
}

function _createTempSelectionChannel(doc) {
  var channel = null;
  try {
    channel = doc.channels.getByName(_TEMP_SELECTION_CHANNEL);
    channel.remove();
  } catch (e) { }

  try {
    channel = doc.channels.add();
    channel.name = _TEMP_SELECTION_CHANNEL;
    doc.selection.store(channel);
    return channel;
  } catch (error) {
    if (channel) {
      try {
        channel.remove();
      } catch (removeError) { }
    }
    return null;
  }
}

function _getAdjustedSelectionBoundsFallback(bounds, amount) {
  if (!bounds || amount === 0) return bounds;
  var delta = Math.abs(amount);
  if (amount < 0) {
    if (bounds.width <= delta * 2 || bounds.height <= delta * 2) {
      return null;
    }
    var contracted = {
      top: bounds.top + delta,
      left: bounds.left + delta,
      right: bounds.right - delta,
      bottom: bounds.bottom - delta,
    };
    contracted.width = contracted.right - contracted.left;
    contracted.height = contracted.bottom - contracted.top;
    contracted.xMid = (contracted.left + contracted.right) / 2;
    contracted.yMid = (contracted.top + contracted.bottom) / 2;
    return contracted;
  } else {
    var expanded = {
      top: Math.max(bounds.top - delta, 0),
      left: Math.max(bounds.left - delta, 0),
      right: bounds.right + delta,
      bottom: bounds.bottom + delta,
    };
    expanded.width = expanded.right - expanded.left;
    expanded.height = expanded.bottom - expanded.top;
    expanded.xMid = (expanded.left + expanded.right) / 2;
    expanded.yMid = (expanded.top + expanded.bottom) / 2;
    return expanded;
  }
}

function _clampAdjustAmount(bounds, amount) {
  if (!bounds || amount >= 0) return amount;
  // Avoid over-contracting small selections: keep at least 2px margin per side
  var maxContract = Math.floor(Math.min(bounds.width, bounds.height) / 2 - 1);
  if (maxContract <= 0) return 0;
  return -Math.min(Math.abs(amount), maxContract);
}

function _getAdjustedSelectionBoundsSequence(bounds, adjustments, preExpandAmount) {
  if (!bounds || !adjustments || !adjustments.length) return bounds;

  var doc;
  try {
    doc = app.activeDocument;
  } catch (error) {
    doc = null;
  }

  if (!doc || !doc.selection) {
    return _getAdjustedSelectionBoundsSequenceFallback(bounds, adjustments);
  }

  var tempChannel = _createTempSelectionChannel(doc);
  if (!tempChannel) {
    return _getAdjustedSelectionBoundsSequenceFallback(bounds, adjustments);
  }

  var adjusted = bounds;
  try {
    // Expand then contract by text size (smooths the selection)
    if (preExpandAmount && preExpandAmount > 0) {
      // First expand
      _modifySelectionBounds(preExpandAmount);
      adjusted = _getCurrentSelectionBounds();
      if (!adjusted) {
        adjusted = bounds;
      }
      // Then contract back by the same amount
      var contractAmount = _clampAdjustAmount(adjusted, -preExpandAmount);
      if (contractAmount !== 0) {
        _modifySelectionBounds(contractAmount);
        adjusted = _getCurrentSelectionBounds();
        if (!adjusted) {
          adjusted = bounds;
        }
      }
    }

    for (var i = 0; i < adjustments.length; i++) {
      var amount = _clampAdjustAmount(adjusted, adjustments[i]);
      if (amount === 0) continue;
      _modifySelectionBounds(amount);
      adjusted = _getCurrentSelectionBounds();
      if (!adjusted) break;
    }
  } catch (error2) {
    adjusted = null;
  } finally {
    try {
      doc.selection.load(tempChannel);
    } catch (restoreError) { }
    try {
      tempChannel.remove();
    } catch (removeError) { }
  }

  if (!adjusted) {
    return _getAdjustedSelectionBoundsSequenceFallback(bounds, adjustments);
  }
  return adjusted;
}

function _getAdjustedSelectionBoundsSequenceFallback(bounds, adjustments) {
  if (!bounds || !adjustments || !adjustments.length) return bounds;
  var current = bounds;
  for (var i = 0; i < adjustments.length; i++) {
    var amount = _clampAdjustAmount(current, adjustments[i]);
    current = _getAdjustedSelectionBoundsFallback(current, amount);
    if (!current) break;
  }
  return current;
}

function _selectionBoundsKey(bounds) {
  if (!bounds) return "";
  return bounds.xMid + "_" + bounds.yMid + "_" + bounds.width + "_" + bounds.height;
}

function _calculateSelectionDimensions(selection, padding) {
  if (!selection) return { width: 0, height: 0 };
  var width = selection.width * _DEFAULT_SELECTION_SCALE;
  var height = selection.height;
  if (padding > 0) {
    var padX = Math.min(padding, Math.max(0, (width - _MIN_TEXTBOX_WIDTH) / 2));
    var padY = Math.min(padding, Math.max(0, (height - _MIN_TEXTBOX_HEIGHT) / 2));
    width = Math.max(width - padX * 2, _MIN_TEXTBOX_WIDTH);
    height = Math.max(height - padY * 2, _MIN_TEXTBOX_HEIGHT);
  }
  return {
    width: width,
    height: height,
  };
}

function _resizeTextBoxToContent(width, currentBounds) {
  var textParams = jamText.getLayerText();
  var textSize = textParams.layerText.textStyleRange[0].textStyle.size;
  _setTextBoxSize(width, currentBounds.height + textSize + 2);
}

function _positionLayerWithinSelection(selection, bounds) {
  if (!selection || !bounds) return;
  var offsetX = selection.xMid - bounds.xMid;
  var offsetY = selection.yMid - bounds.yMid;
  _moveLayer(offsetX, offsetY);
}

function _createMagicWandSelection(tolerance) {
  try {
    var bounds = _getCurrentTextLayerBounds();
    var x = Math.max(bounds.left - 5, 0);
    var y = Math.max(bounds.yMid, 0);
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putProperty(charID.Channel, charID.FrameSelect);
    desc.putReference(charID.Null, ref);

    var pos = new ActionDescriptor();
    pos.putUnitDouble(charID.Horizontal, charID.PixelUnit, x);
    pos.putUnitDouble(charID.Vertical, charID.PixelUnit, y);
    desc.putObject(charID.To, stringIDToTypeID("paint"), pos);

    desc.putInteger(stringIDToTypeID("tolerance"), tolerance || 20);
    desc.putBoolean(stringIDToTypeID("merged"), true);
    desc.putBoolean(stringIDToTypeID("antiAlias"), true);
    executeAction(charID.Set, desc, DialogModes.NO);
  } catch (e) { }
}

function _moveLayer(offsetX, offsetY) {
  var amount = new ActionDescriptor();
  amount.putUnitDouble(charID.Horizontal, charID.PixelUnit, offsetX);
  amount.putUnitDouble(charID.Vertical, charID.PixelUnit, offsetY);
  var target = new ActionDescriptor();
  target.putReference(charID.Null, _createCurrent(charID.Layer));
  target.putObject(charID.To, charID.Offset, amount);
  executeAction(charID.Move, target, DialogModes.NO);
}

/**
 * Retrieve stroke information from the active layer.
 * Returns null if no stroke is found.
 */
function _getLayerStroke() {
  var ref = new ActionReference();
  ref.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("Lefx"));
  ref.putEnumerated(charIDToTypeID("Lyr "), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
  var desc = executeActionGet(ref);
  if (!desc.hasKey(charIDToTypeID("Lefx"))) return null;

  var fx = desc.getObjectValue(charIDToTypeID("Lefx"));
  if (!fx.hasKey(charIDToTypeID("FrFX"))) return null;

  var fr = fx.getObjectValue(charIDToTypeID("FrFX"));
  var col = fr.getObjectValue(charIDToTypeID("Clr "));

  return {
    enabled: fr.getBoolean(charIDToTypeID("enab")),
    position: fr.getEnumerationValue(charIDToTypeID("Styl")) == charIDToTypeID("OutF") ? "outer" : "other",
    size: fr.getUnitDoubleValue(charIDToTypeID("Sz  ")),
    opacity: fr.getUnitDoubleValue(charIDToTypeID("Opct")),
    color: {
      r: col.getDouble(charIDToTypeID("Rd  ")),
      g: col.getDouble(charIDToTypeID("Grn ")),
      b: col.getDouble(charIDToTypeID("Bl  ")),
    },
  };
}

/**
 * Apply or update a stroke on the active layer.
 * @param {Object} stroke - {size, color:{r,g,b}, opacity, enabled}
 *                          position is forced to "outer".
 */
function _setLayerStroke(stroke) {
  if (!stroke || (stroke.size <= 0 && stroke.enabled !== true)) return;

  var d = new ActionDescriptor();
  var r = new ActionReference();
  r.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("Lefx"));
  r.putEnumerated(charIDToTypeID("Lyr "), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
  d.putReference(charIDToTypeID("null"), r);

  var fx = new ActionDescriptor();
  fx.putUnitDouble(charIDToTypeID("Scl "), charIDToTypeID("#Prc"), 100);

  var fr = new ActionDescriptor();
  fr.putBoolean(charIDToTypeID("enab"), true);
  fr.putBoolean(stringIDToTypeID("present"), true);
  fr.putBoolean(stringIDToTypeID("showInDialog"), true);

  fr.putEnumerated(charIDToTypeID("Styl"), charIDToTypeID("FStl"), charIDToTypeID("OutF"));
  fr.putEnumerated(charIDToTypeID("PntT"), charIDToTypeID("FrFl"), charIDToTypeID("SClr"));
  fr.putEnumerated(charIDToTypeID("Md  "), charIDToTypeID("BlnM"), charIDToTypeID("Nrml"));

  fr.putUnitDouble(charIDToTypeID("Sz  "), charIDToTypeID("#Pxl"), stroke.size || 3);
  fr.putUnitDouble(charIDToTypeID("Opct"), charIDToTypeID("#Prc"), stroke.opacity || 100);

  var c = new ActionDescriptor();
  c.putDouble(charIDToTypeID("Rd  "), stroke.color.r);
  c.putDouble(charIDToTypeID("Grn "), stroke.color.g);
  c.putDouble(charIDToTypeID("Bl  "), stroke.color.b);
  fr.putObject(charIDToTypeID("Clr "), charIDToTypeID("RGBC"), c);

  fx.putObject(charIDToTypeID("FrFX"), charIDToTypeID("FrFX"), fr);
  d.putObject(charIDToTypeID("T   "), charIDToTypeID("Lefx"), fx);

  executeAction(charIDToTypeID("setd"), d, DialogModes.NO);
}

function _setDiacXOffset(val) {
  var d = new ActionDescriptor();
  var r = new ActionReference();
  r.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("TxtS"));
  r.putEnumerated(charIDToTypeID("TxLr"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
  d.putReference(charIDToTypeID("null"), r);

  var t = new ActionDescriptor();
  t.putInteger(stringIDToTypeID("textOverrideFeatureName"), 808466486);
  t.putInteger(stringIDToTypeID("typeStyleOperationType"), 3);
  t.putUnitDouble(stringIDToTypeID("diacXOffset"), charIDToTypeID("#Pxl"), val);
  d.putObject(charIDToTypeID("T   "), charIDToTypeID("TxtS"), t);

  executeAction(charIDToTypeID("setd"), d, DialogModes.NO);
}

function _setMarkYOffset(val) {
  var d = new ActionDescriptor();
  var r = new ActionReference();
  r.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("TxtS"));
  r.putEnumerated(charIDToTypeID("TxLr"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
  d.putReference(charIDToTypeID("null"), r);

  var t = new ActionDescriptor();
  t.putInteger(stringIDToTypeID("textOverrideFeatureName"), 808466488);
  t.putInteger(stringIDToTypeID("typeStyleOperationType"), 3);
  t.putUnitDouble(stringIDToTypeID("markYDistFromBaseline"), charIDToTypeID("#Pxl"), val);
  d.putObject(charIDToTypeID("T   "), charIDToTypeID("TxtS"), t);

  executeAction(charIDToTypeID("setd"), d, DialogModes.NO);
}

function _applyMiddleEast(textStyle) {
  if (!textStyle) return;
  if (textStyle.diacXOffset != null) _setDiacXOffset(textStyle.diacXOffset);
  if (textStyle.markYDistFromBaseline != null) _setMarkYOffset(textStyle.markYDistFromBaseline);
}

function _applyTextDirection(direction, textLength) {
  if (!direction) return;
  var psDirection = direction === "rtl" ? "dirRightToLeft" : "dirLeftToRight";

  try {
    var currentText = jamText.getLayerText();
    if (
      !currentText ||
      !currentText.layerText ||
      !currentText.layerText.paragraphStyleRange ||
      !currentText.layerText.paragraphStyleRange.length
    ) {
      return;
    }

    var updatedText = _clone(currentText);
    var paragraphRanges = updatedText.layerText.paragraphStyleRange;
    var targetLength = textLength;
    if (targetLength == null && updatedText.layerText && updatedText.layerText.textKey) {
      targetLength = updatedText.layerText.textKey.length;
    }

    for (var i = 0; i < paragraphRanges.length; i++) {
      var range = paragraphRanges[i] || {};
      var paragraphStyle = range.paragraphStyle || {};

      paragraphStyle.directionType = psDirection;
      paragraphStyle.textComposerEngine = "textOptycaComposer";

      range.paragraphStyle = paragraphStyle;
      if (targetLength != null) {
        range.from = typeof range.from === "number" ? range.from : 0;
        range.to = targetLength;
      }
      paragraphRanges[i] = range;
    }

    updatedText.layerText.paragraphStyleRange = paragraphRanges;
    jamText.setLayerText(updatedText);
  } catch (e) {
    // Ignore errors if directionType is not supported on this PS version
  }
}

function _createAndSetLayerText(data, width, height) {
  var style = _ensureStyle(data.style);
  style.textProps.layerText.textKey = data.text.replace(/\n+/g, "");
  style.textProps.layerText.textStyleRange[0].to = data.text.length;
  style.textProps.layerText.paragraphStyleRange[0].to = data.text.length;
  var sizeProp = style.textProps.layerText.textStyleRange[0].textStyle.size;
  if (typeof sizeProp !== "number") {
    try {
      var textParams = jamText.getLayerText();
      _hostState.fallbackTextSize = textParams.layerText.textStyleRange[0].textStyle.size;
    } catch (error) { }
    style.textProps.layerText.textStyleRange[0].textStyle.size = _hostState.fallbackTextSize;
  }
  style.textProps.layerText.textShape = [
    {
      textType: "box",
      orientation: "horizontal",
      bounds: {
        top: 0,
        left: 0,
        right: _convertPixelToPoint(width),
        bottom: _convertPixelToPoint(height),
      },
    },
  ];
  jamEngine.jsonPlay("make", {
    target: ["<reference>", [["textLayer", ["<class>", null]]]],
    using: jamText.toLayerTextObject(style.textProps),
  });
  _applyMiddleEast(style.textProps.layerText.textStyleRange[0].textStyle);
  if (style.stroke) {
    _setLayerStroke(style.stroke);
  }
  // Apply text direction if specified
  if (data.direction) {
    _applyTextDirection(data.direction, data.text.length);
  }
}

function _setTextBoxSize(width, height) {
  var box = [
    {
      textType: "box",
      orientation: "horizontal",
      bounds: {
        top: 0,
        left: 0,
        right: _convertPixelToPoint(width),
        bottom: _convertPixelToPoint(height),
      },
    },
  ];
  jamText.setLayerText({ layerText: { textShape: box } });
}

function _checkSelection(options) {
  var selection = _getCurrentSelectionBounds();
  if (selection === undefined) {
    return { error: "noSelection" };
  }

  var adjustAmount = 0;
  var adjustSequence = null;
  var preExpandAmount = 0;
  if (options && options.adjustAmount !== undefined) {
    adjustAmount = options.adjustAmount;
  }
  if (options && options.adjustSequence && options.adjustSequence.length) {
    adjustSequence = options.adjustSequence;
  }
  if (options && options.preExpandAmount !== undefined) {
    preExpandAmount = options.preExpandAmount;
  }

  var adjustedSelection = selection;
  if (adjustSequence) {
    adjustedSelection = _getAdjustedSelectionBoundsSequence(selection, adjustSequence, preExpandAmount);
  } else if (adjustAmount !== 0) {
    adjustedSelection = _getAdjustedSelectionBounds(selection, adjustAmount);
  }
  if (!adjustedSelection || adjustedSelection.width * adjustedSelection.height < 200) {
    return { error: "smallSelection" };
  }

  return adjustedSelection;
}

function _forEachSelectedLayer(action) {
  var selectedLayers = [];
  var reference = new ActionReference();
  var targetLayers = stringIDToTypeID("targetLayers");
  reference.putProperty(charID.Property, targetLayers);
  reference.putEnumerated(charID.Document, charID.Ordinal, charID.Target);
  var doc = executeActionGet(reference);
  if (doc.hasKey(targetLayers)) {
    doc = doc.getList(targetLayers);
    var ref2 = new ActionReference();
    ref2.putProperty(charID.Property, charID.Background);
    ref2.putEnumerated(charID.Layer, charID.Ordinal, charID.Back);
    var offset = executeActionGet(ref2).getBoolean(charID.Background) ? 0 : 1;
    for (var i = 0; i < doc.count; i++) {
      selectedLayers.push(doc.getReference(i).getIndex() + offset);
    }
  }
  if (selectedLayers.length > 1) {
    for (var j = 0; j < selectedLayers.length; j++) {
      var descr = new ActionDescriptor();
      var ref3 = new ActionReference();
      ref3.putIndex(charID.Layer, selectedLayers[j]);
      descr.putReference(charID.Null, ref3);
      executeAction(charID.Select, descr, DialogModes.NO);
      action(selectedLayers[j]);
    }
    var ref4 = new ActionReference();
    for (var k = 0; k < selectedLayers.length; k++) {
      ref4.putIndex(charID.Layer, selectedLayers[k]);
    }
    var descr2 = new ActionDescriptor();
    descr2.putReference(charID.Null, ref4);
    executeAction(charID.Select, descr2, DialogModes.NO);
  } else if (selectedLayers.length === 1) {
    action(selectedLayers[0]);
  }
}

/* ========================================================= */
/* ============ full methods for suspendHistory ============ */
/* ========================================================= */

function _setActiveLayerText() {
  var state = _hostState.setActiveLayerText;
  var payload = state.data;
  state.result = "";
  if (!payload) {
    return;
  } else if (!documents.length) {
    state.result = "doc";
    return;
  } else if (!_layerIsTextLayer()) {
    state.result = "layer";
    return;
  }
  var dataText = payload.text;
  var dataStyle = payload.style;
  var targetTextLength = 0;

  _forEachSelectedLayer(function () {
    var oldBounds = _getCurrentTextLayerBounds();
    var isPoint = _textLayerIsPointText();
    if (isPoint) _changeToBoxText();
    var oldTextParams = jamText.getLayerText();
    var newTextParams;
    if (dataText && dataStyle) {
      newTextParams = dataStyle.textProps;
      if (newTextParams.layerText.textStyleRange[0].textStyle.size == null &&
        oldTextParams.layerText.textStyleRange &&
        oldTextParams.layerText.textStyleRange[0] &&
        oldTextParams.layerText.textStyleRange[0].textStyle.size != null) {
        newTextParams.layerText.textStyleRange[0].textStyle.size = oldTextParams.layerText.textStyleRange[0].textStyle.size;
      }
      newTextParams.layerText.textKey = dataText.replace(/\n+/g, "");
      newTextParams.layerText.textStyleRange[0].to = dataText.length;
      newTextParams.layerText.paragraphStyleRange[0].to = dataText.length;
      targetTextLength = dataText.length;
    } else if (dataText) {
      newTextParams = {
        layerText: {
          textKey: dataText.replace(/\n+/g, ""),
        },
      };
      if (oldTextParams.layerText.textStyleRange && oldTextParams.layerText.textStyleRange[0]) {
        newTextParams.layerText.textStyleRange = [oldTextParams.layerText.textStyleRange[0]];
        newTextParams.layerText.textStyleRange[0].to = dataText.length;
      }
      if (oldTextParams.layerText.paragraphStyleRange && oldTextParams.layerText.paragraphStyleRange[0]) {
        // Create a minimal paragraphStyleRange without directionType to avoid RTL issues
        var oldParagraphStyle = oldTextParams.layerText.paragraphStyleRange[0].paragraphStyle || {};
        var newParagraphStyle = {};

        // Copy only safe properties, explicitly excluding directionType
        for (var i = 0; i < _SAFE_PARAGRAPH_PROPS.length; i++) {
          var prop = _SAFE_PARAGRAPH_PROPS[i];
          if (oldParagraphStyle[prop] !== undefined) {
            newParagraphStyle[prop] = oldParagraphStyle[prop];
          }
        }

        newTextParams.layerText.paragraphStyleRange = [{
          from: 0,
          to: dataText.length,
          paragraphStyle: newParagraphStyle
        }];
      }
      targetTextLength = dataText.length;
    } else if (dataStyle) {
      var text = oldTextParams.layerText.textKey || "";
      newTextParams = dataStyle.textProps;
      newTextParams.layerText.textStyleRange[0].to = text.length;
      newTextParams.layerText.paragraphStyleRange[0].to = text.length;
      targetTextLength = text.length;
    }
    var retainedShape = oldTextParams.layerText.textShape && oldTextParams.layerText.textShape[0];
    if (isPoint && retainedShape && retainedShape.bounds) {
      var oldTextStyle = oldTextParams.layerText.textStyleRange &&
        oldTextParams.layerText.textStyleRange[0] &&
        oldTextParams.layerText.textStyleRange[0].textStyle;
      var styleTextStyle = dataStyle &&
        dataStyle.textProps &&
        dataStyle.textProps.layerText &&
        dataStyle.textProps.layerText.textStyleRange &&
        dataStyle.textProps.layerText.textStyleRange[0] &&
        dataStyle.textProps.layerText.textStyleRange[0].textStyle;
      var oldSize = oldTextStyle && oldTextStyle.size;
      var newSize = styleTextStyle && styleTextStyle.size != null ? styleTextStyle.size : oldSize;
      var widthScale = oldSize && newSize ? newSize / oldSize : 1;
      if (!(widthScale > 0)) widthScale = 1;
      if (widthScale < 1) widthScale = 1;
      var bounds = retainedShape.bounds;
      var currentWidth = bounds.right - bounds.left;
      var currentHeight = bounds.bottom - bounds.top;
      var oldWidthPoints = typeof oldBounds.width === "number" ? _convertPixelToPoint(oldBounds.width) : currentWidth;
      var oldHeightPoints = typeof oldBounds.height === "number" ? _convertPixelToPoint(oldBounds.height) : currentHeight;
      var targetWidth = currentWidth * widthScale;
      var targetHeight = currentHeight * widthScale;
      if (targetWidth < oldWidthPoints * widthScale) targetWidth = oldWidthPoints * widthScale;
      var minWidthPadding = (newSize || oldSize || 12) * 0.5;
      if (targetWidth < oldWidthPoints + minWidthPadding) targetWidth = oldWidthPoints + minWidthPadding;
      var minHeightPadding = (newSize || oldSize || 12) * 0.75;
      if (targetHeight < oldHeightPoints * widthScale) targetHeight = oldHeightPoints * widthScale;
      if (targetHeight < oldHeightPoints + minHeightPadding) targetHeight = oldHeightPoints + minHeightPadding;
      bounds.right = bounds.left + targetWidth;
      bounds.bottom = bounds.top + targetHeight;
    }
    newTextParams.layerText.antiAlias = oldTextParams.layerText.antiAlias || "antiAliasSmooth";
    if (retainedShape) {
      newTextParams.layerText.textShape = [retainedShape];
    }
    newTextParams.typeUnit = oldTextParams.typeUnit;
    jamText.setLayerText(newTextParams);
    var userDirection = payload.direction;
    if (userDirection === "") userDirection = null;
    _applyTextDirection(userDirection, targetTextLength);
    _applyMiddleEast(newTextParams.layerText.textStyleRange[0].textStyle);
    if (dataStyle && dataStyle.stroke) {
      _setLayerStroke(dataStyle.stroke);
    }
    var newBounds = _getCurrentTextLayerBounds();
    if (isPoint) {
      _changeToPointText();
    } else {
      var textSize = 12;
      var styleSize = dataStyle && dataStyle.textProps.layerText.textStyleRange[0].textStyle.size;
      if (styleSize != null) {
        textSize = styleSize;
      } else if (oldTextParams.layerText.textStyleRange && oldTextParams.layerText.textStyleRange[0] && oldTextParams.layerText.textStyleRange[0].textStyle.size != null) {
        textSize = oldTextParams.layerText.textStyleRange[0].textStyle.size;
      }
      newTextParams.layerText.textShape[0].bounds.bottom = _convertPixelToPoint(newBounds.height + textSize + 2);
      jamText.setLayerText({
        layerText: {
          textShape: newTextParams.layerText.textShape,
        },
      });
    }
    newBounds = _getCurrentTextLayerBounds();
    if (!oldBounds.bottom) oldBounds = newBounds;
    var offsetX = oldBounds.xMid - newBounds.xMid;
    var offsetY = oldBounds.yMid - newBounds.yMid;
    _moveLayer(offsetX, offsetY);
  });

  state.result = "";
}

function _createTextLayerInSelection() {
  var state = _hostState.createTextLayerInSelection;
  if (!documents.length) {
    state.result = "doc";
    return;
  }

  // Get the text size from the style to pre-expand/dilate selection
  var textSize = _hostState.fallbackTextSize || 20;
  var style = _ensureStyle(state.data.style);
  if (style && style.textProps && style.textProps.layerText &&
    style.textProps.layerText.textStyleRange &&
    style.textProps.layerText.textStyleRange[0] &&
    style.textProps.layerText.textStyleRange[0].textStyle &&
    style.textProps.layerText.textStyleRange[0].textStyle.size) {
    textSize = style.textProps.layerText.textStyleRange[0].textStyle.size;
  }

  var selection = _checkSelection({
    adjustSequence: _DEFAULT_ADJUST_SEQUENCE,
    preExpandAmount: textSize
  });
  if (selection.error) {
    state.result = selection.error;
    return;
  }
  var dimensions = _calculateSelectionDimensions(selection, state.padding);
  _createAndSetLayerText(state.data, dimensions.width, dimensions.height);
  var bounds = _getCurrentTextLayerBounds();
  if (state.point) {
    _changeToPointText();
  } else {
    _resizeTextBoxToContent(dimensions.width, bounds);
  }
  bounds = _getCurrentTextLayerBounds();
  _positionLayerWithinSelection(selection, bounds);
  state.result = "";
}

function _alignTextLayerToSelection() {
  var state = _hostState.alignTextLayerToSelection;
  if (!documents.length) {
    state.result = "doc";
    return;
  } else if (!_layerIsTextLayer()) {
    state.result = "layer";
    return;
  }

  // Get the text size to pre-expand/dilate selection
  var textSize = _getTextLayerSize();

  var selection = _checkSelection({
    adjustSequence: _DEFAULT_ADJUST_SEQUENCE,
    preExpandAmount: textSize
  });
  if (selection.error) {
    if (selection.error === "noSelection") {
      _createMagicWandSelection(20);
      selection = _checkSelection({
        adjustSequence: _DEFAULT_ADJUST_SEQUENCE,
        preExpandAmount: textSize
      });
    }
    if (selection.error) {
      state.result = selection.error;
      return;
    }
  }
  var wasPoint = _textLayerIsPointText();
  var bounds = _getCurrentTextLayerBounds();

  if (state.resize && !wasPoint) {
    var dimensions = _calculateSelectionDimensions(selection, state.padding);
    _setTextBoxSize(dimensions.width, dimensions.height);
    var textBounds = _getCurrentTextLayerBounds();
    _resizeTextBoxToContent(dimensions.width, textBounds);
    bounds = _getCurrentTextLayerBounds();
  }

  _deselect();
  _positionLayerWithinSelection(selection, bounds);
  if (wasPoint) {
    _changeToPointText();
  }
  state.result = "";
}

function _alignTextLayerToTarget() {
  var state = _hostState.alignTextLayerToTarget;
  if (!documents.length) {
    state.result = "doc";
    return;
  } else if (!_layerIsTextLayer()) {
    state.result = "layer";
    return;
  } else if (!state.target || state.target.xMid === undefined || state.target.yMid === undefined) {
    state.result = "target";
    return;
  }

  var wasPoint = _textLayerIsPointText();
  var bounds = _getCurrentTextLayerBounds();

  if (state.resize && !wasPoint) {
    var dimensions = _calculateSelectionDimensions(state.target, state.padding);
    _setTextBoxSize(dimensions.width, dimensions.height);
    var textBounds = _getCurrentTextLayerBounds();
    _resizeTextBoxToContent(dimensions.width, textBounds);
    bounds = _getCurrentTextLayerBounds();
  }

  _deselect();
  _positionLayerWithinSelection(state.target, bounds);
  if (wasPoint) {
    _changeToPointText();
  }
  state.result = "";
}

function _changeActiveLayerTextSize() {
  var state = _hostState.changeActiveLayerTextSize;
  if (!documents.length) {
    state.result = "doc";
    return;
  } else if (!_layerIsTextLayer()) {
    state.result = "layer";
    return;
  } else if (!state.value) {
    state.result = "";
    return;
  }

  // Version optimisée utilisant les actions Photoshop directes
  _forEachSelectedLayer(function () {
    try {
      // Utiliser la méthode rapide d'actions Photoshop pour changer la taille
      var ref = new ActionReference();
      ref.putProperty(charID.Property, charID.TextStyle);
      ref.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);

      var currentTextStyle = executeActionGet(ref);
      if (currentTextStyle.hasKey(charID.TextStyle)) {
        var textStyle = currentTextStyle.getObjectValue(charID.TextStyle);
        var currentSize = textStyle.getDouble(charID.Size);
        var sizeUnit = textStyle.getUnitDoubleType(charID.Size);
        var newSize = currentSize + state.value;

        // Appliquer le nouveau size directement
        var descriptor = new ActionDescriptor();
        var reference = new ActionReference();
        reference.putProperty(charID.Property, charID.TextStyle);
        reference.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);
        descriptor.putReference(charID.Null, reference);

        var newTextStyle = new ActionDescriptor();
        newTextStyle.putUnitDouble(charID.Size, sizeUnit, newSize);
        descriptor.putObject(charID.To, charID.TextStyle, newTextStyle);

        executeAction(charID.Set, descriptor, DialogModes.NO);
      }
    } catch (e) {
      // Si la méthode rapide échoue, utiliser l'ancienne méthode
      var oldTextParams = jamText.getLayerText();
      var text = oldTextParams.layerText.textKey.replace(/\n+/g, "");
      if (!text) {
        state.result = "layer";
        return;
      }
      var oldBounds = _getCurrentTextLayerBounds();
      var isPoint = _textLayerIsPointText();
      var newTextParams = {
        typeUnit: oldTextParams.typeUnit,
        layerText: {
          textKey: text,
          textGridding: oldTextParams.layerText.textGridding || "none",
          orientation: oldTextParams.layerText.orientation || "horizontal",
          antiAlias: oldTextParams.layerText.antiAlias || "antiAliasSmooth",
          textStyleRange: [oldTextParams.layerText.textStyleRange[0]],
        },
      };
      if (oldTextParams.layerText.paragraphStyleRange) {
        var oldParStyle = oldTextParams.layerText.paragraphStyleRange[0].paragraphStyle;
        newTextParams.layerText.paragraphStyleRange = [oldTextParams.layerText.paragraphStyleRange[0]];
        newTextParams.layerText.paragraphStyleRange[0].paragraphStyle.textEveryLineComposer = oldParStyle.textEveryLineComposer || false;
        newTextParams.layerText.paragraphStyleRange[0].paragraphStyle.burasagari = oldParStyle.burasagari || "burasagariNone";
        newTextParams.layerText.paragraphStyleRange[0].to = text.length;
      }
      var oldSize = newTextParams.layerText.textStyleRange[0].textStyle.size;
      var newTextSize = oldSize + state.value;
      newTextParams.layerText.textStyleRange[0].textStyle.size = newTextSize;

      // Ajuster l'interligne
      var textStyle = newTextParams.layerText.textStyleRange[0].textStyle;
      if (textStyle.autoLeading || textStyle.leading === undefined) {
        // Si l'interligne est en auto, on le laisse en auto
        textStyle.autoLeading = true;
        // On supprime la propriété leading si elle existe pour s'assurer que l'auto soit appliqué
        delete textStyle.leading;
      } else {
        // Sinon, on ajuste l'interligne de la même valeur que la taille du texte
        var oldLeading = textStyle.leading;
        var newLeading = oldLeading + state.value;
        textStyle.leading = newLeading;
        textStyle.autoLeading = false;
      }

      newTextParams.layerText.textStyleRange[0].to = text.length;
      if (!isPoint) {
        var ratio = newTextSize / oldSize;
        newTextParams.layerText.textShape = [oldTextParams.layerText.textShape[0]];
        var shapeBounds = newTextParams.layerText.textShape[0].bounds;
        shapeBounds.top *= ratio;
        shapeBounds.left *= ratio;
        shapeBounds.bottom *= ratio;
        shapeBounds.right *= ratio;
      }
      jamText.setLayerText(newTextParams);
      _applyMiddleEast(newTextParams.layerText.textStyleRange[0].textStyle);
      var newBounds = _getCurrentTextLayerBounds();
      var offsetX = oldBounds.xMid - newBounds.xMid;
      var offsetY = oldBounds.yMid - newBounds.yMid;
      _moveLayer(offsetX, offsetY);
    }
  });

  state.result = "";
}

function _changeSize_alt() {
  var increasing = _hostState.changeActiveLayerTextSize.value > 0;
  _forEachSelectedLayer(function () {
    var a = new ActionReference();
    a.putProperty(charID.Property, charID.Text);
    a.putEnumerated(charID.Layer, charID.Ordinal, charID.Target);
    var currentLayer = executeActionGet(a);
    if (currentLayer.hasKey(charID.Text)) {
      var settings = currentLayer.getObjectValue(charID.Text);
      var textStyleRange = settings.getList(charID.TextStyleRange);
      var sizes = [];
      var units = [];
      var proceed = true;
      for (var i = 0; i < textStyleRange.count; i++) {
        var style = textStyleRange.getObjectValue(i).getObjectValue(charID.TextStyle);
        sizes[i] = style.getDouble(charID.Size);
        units[i] = style.getUnitDoubleType(charID.Size);
        if (i > 0 && (sizes[i] !== sizes[i - 1] || units[i] !== units[i - 1])) {
          proceed = false;
          break;
        }
      }
      var amount = 0.2; // mm
      if (units[0] === charID.PixelUnit) amount = 1; // pixel
      else if (units[0] === 592473716) amount = 0.5; // point
      if (!increasing) amount *= -1;
      if (proceed) {
        var aa = new ActionDescriptor();
        var d = new ActionReference();
        d.putProperty(charID.Property, charID.TextStyle);
        d.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);
        aa.putReference(charID.Null, d);
        var e = new ActionDescriptor();
        e.putUnitDouble(charID.Size, units[0], sizes[0] + amount);
        aa.putObject(charID.To, charID.TextStyle, e);
        executeAction(charID.Set, aa, DialogModes.NO);
      }
    }
  });
  _hostState.changeActiveLayerTextSize.result = "";
}

/* ======================================================== */
/* ==================== public methods ==================== */
/* ======================================================== */

function nativeAlert(data) {
  if (!data) return "";
  alert(data.text, data.title, data.isError);
}

function nativeConfirm(data) {
  if (!data) return "";
  var result = confirm(data.text, false, data.title);
  return result ? "1" : "";
}

function getUserFonts() {
  var fontsArr = [];
  for (var i = 0; i < app.fonts.length; i++) {
    var font = app.fonts[i];
    fontsArr.push({
      name: font.name,
      postScriptName: font.postScriptName,
      family: font.family,
      style: font.style,
    });
  }
  return jamJSON.stringify({
    fonts: fontsArr,
  });
}

function getHotkeyPressed() {
  try {
    var state = ScriptUI.environment && ScriptUI.environment.keyboardState;
    var string = "a";

    if (state && state.metaKey) {
      string += "WINa";
    }
    if (state && state.ctrlKey) {
      string += "CTRLa";
    }
    if (state && state.altKey) {
      string += "ALTa";
    }
    if (state && state.shiftKey) {
      string += "SHIFTa";
    }
    if (state && state.keyName) {
      string += state.keyName.toUpperCase() + "a";
    }
    return string;
  } catch (e) {
    return "a";
  }
}

function getActiveLayerText() {
  if (!documents.length) {
    return "";
  } else if (activeDocument.activeLayer.kind != LayerKind.TEXT) {
    return "";
  }
  return jamJSON.stringify({
    textProps: jamText.getLayerText(),
    stroke: _getLayerStroke(),
  });
}

function getActiveLayerBounds() {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  try {
    var boundsTypeId = stringIDToTypeID("bounds");
    var bounds = _getCurrent(charID.Layer, boundsTypeId).getObjectValue(boundsTypeId);
    return jamJSON.stringify(_getBoundsFromDescriptor(bounds));
  } catch (e) {
    return jamJSON.stringify({ error: "bounds" });
  }
}

function getActiveTextBoxBounds() {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  if (!_layerIsTextLayer()) {
    return jamJSON.stringify({ error: "layer" });
  }
  try {
    var textParams = jamText.getLayerText();
    var textShape = textParams.layerText.textShape;
    if (!textShape || !textShape[0] || !textShape[0].bounds) {
      // Fallback to layer bounds if no textShape (point text)
      var boundsTypeId = stringIDToTypeID("bounds");
      var bounds = _getCurrent(charID.Layer, boundsTypeId).getObjectValue(boundsTypeId);
      return jamJSON.stringify(_getBoundsFromDescriptor(bounds));
    }
    var shapeBounds = textShape[0].bounds;
    // textShape bounds are in points, convert to pixels
    var resolution = activeDocument.resolution;
    var pointToPixel = resolution / 72;

    // Get layer position to calculate absolute coordinates
    var boundsTypeId = stringIDToTypeID("bounds");
    var layerBounds = _getCurrent(charID.Layer, boundsTypeId).getObjectValue(boundsTypeId);
    var layerInfo = _getBoundsFromDescriptor(layerBounds);

    // textShape bounds are relative (top/left usually 0), width/height matter
    var width = (shapeBounds.right - shapeBounds.left) * pointToPixel;
    var height = (shapeBounds.bottom - shapeBounds.top) * pointToPixel;

    return jamJSON.stringify({
      left: layerInfo.left,
      top: layerInfo.top,
      right: layerInfo.left + width,
      bottom: layerInfo.top + height,
      width: width,
      height: height,
      xMid: layerInfo.left + width / 2,
      yMid: layerInfo.top + height / 2
    });
  } catch (e) {
    return jamJSON.stringify({ error: "textbox", detail: "" + e });
  }
}

function getActiveDocumentInfo() {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var startUnits = app.preferences.rulerUnits;
  try {
    app.preferences.rulerUnits = Units.PIXELS;
    var doc = app.activeDocument;
    return jamJSON.stringify({
      width: doc.width.value,
      height: doc.height.value,
      resolution: doc.resolution,
    });
  } catch (e) {
    return jamJSON.stringify({ error: "docInfo" });
  } finally {
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function exportAiRoi(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  if (!data || !data.path) {
    return jamJSON.stringify({ error: "path" });
  }

  var startUnits = app.preferences.rulerUnits;
  var startDialogs = app.displayDialogs;
  var tmpDoc = null;
  try {
    app.preferences.rulerUnits = Units.PIXELS;
    try {
      app.displayDialogs = DialogModes.NO;
    } catch (eDialog) { }

    var doc = app.activeDocument;
    var docW = doc.width.value;
    var docH = doc.height.value;

    var left = Math.max(0, Math.floor(data.left || 0));
    var top = Math.max(0, Math.floor(data.top || 0));
    var right = Math.min(docW, Math.ceil(data.right || 0));
    var bottom = Math.min(docH, Math.ceil(data.bottom || 0));

    if (right <= left || bottom <= top) {
      return jamJSON.stringify({ error: "roi" });
    }

    tmpDoc = doc.duplicate("__TypeR_AI_ROI__", true);
    try {
      // Ensure RGB for the detector.
      tmpDoc.changeMode(ChangeMode.RGB);
    } catch (eMode) { }

    tmpDoc.crop([left, top, right, bottom]);
    try {
      tmpDoc.flatten();
    } catch (eFlat) { }

    var file = new File(data.path);
    var pngOpts = new PNGSaveOptions();
    pngOpts.interlaced = false;
    tmpDoc.saveAs(file, pngOpts, true, Extension.LOWERCASE);
    tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
    tmpDoc = null;

    return jamJSON.stringify({
      path: data.path,
      width: right - left,
      height: bottom - top,
      left: left,
      top: top,
      right: right,
      bottom: bottom,
    });
  } catch (e) {
    return jamJSON.stringify({ error: "export", detail: "" + e });
  } finally {
    try {
      if (tmpDoc) {
        tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
      }
    } catch (eClose2) { }
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function exportActiveDocPng(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  if (!data || !data.path) {
    return jamJSON.stringify({ error: "path" });
  }

  var startUnits = app.preferences.rulerUnits;
  var tmpDoc = null;
  try {
    app.preferences.rulerUnits = Units.PIXELS;

    var doc = app.activeDocument;
    var docW = doc.width.value;
    var docH = doc.height.value;

    tmpDoc = doc.duplicate("__TypeR_EXPORT__", true);
    try {
      tmpDoc.changeMode(ChangeMode.RGB);
    } catch (eMode) { }
    try {
      tmpDoc.flatten();
    } catch (eFlat) { }

    var file = new File(data.path);
    var pngOpts = new PNGSaveOptions();
    pngOpts.interlaced = false;
    tmpDoc.saveAs(file, pngOpts, true, Extension.LOWERCASE);
    tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
    tmpDoc = null;

    return jamJSON.stringify({
      path: data.path,
      width: docW,
      height: docH,
      resolution: doc.resolution,
    });
  } catch (e) {
    return jamJSON.stringify({ error: "export", detail: "" + e });
  } finally {
    try {
      if (tmpDoc) {
        tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
      }
    } catch (eClose2) { }
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function _findLayerSetByName(parent, name) {
  if (!parent || !name) return null;
  var sets = parent.layerSets || [];
  for (var i = 0; i < sets.length; i++) {
    var s = sets[i];
    if (s && s.name === name) return s;
    var inner = _findLayerSetByName(s, name);
    if (inner) return inner;
  }
  return null;
}

function _collectArtLayers(parent, out) {
  if (!parent || !out) return;
  var layers = parent.artLayers || [];
  for (var i = 0; i < layers.length; i++) {
    out.push(layers[i]);
  }
  var sets = parent.layerSets || [];
  for (var j = 0; j < sets.length; j++) {
    _collectArtLayers(sets[j], out);
  }
}

function _unitToPx(v) {
  try {
    return v.as("px");
  } catch (e) { }
  try {
    return v.value;
  } catch (e2) { }
  return Number(v) || 0;
}

function _getLayerBoundsPx(layer, useNoEffects) {
  var b = layer.bounds;
  if (useNoEffects) {
    try {
      if (layer.boundsNoEffects) b = layer.boundsNoEffects;
    } catch (eNoEff) { }
  }
  var left = _unitToPx(b[0]);
  var top = _unitToPx(b[1]);
  var right = _unitToPx(b[2]);
  var bottom = _unitToPx(b[3]);
  return {
    left: left,
    top: top,
    right: right,
    bottom: bottom,
    width: right - left,
    height: bottom - top,
    xMid: (left + right) / 2,
    yMid: (top + bottom) / 2,
  };
}

function _median(arr) {
  if (!arr || !arr.length) return 0;
  var copy = arr.slice(0);
  copy.sort(function (a, b) {
    return a - b;
  });
  var mid = Math.floor(copy.length / 2);
  if (copy.length % 2) return copy[mid];
  return (copy[mid - 1] + copy[mid]) / 2;
}

function _padNum(num, width) {
  var s = "" + num;
  while (s.length < width) s = "0" + s;
  return s;
}

function _createRectShapeLayer(left, top, right, bottom, color) {
  var desc = new ActionDescriptor();
  var ref = new ActionReference();
  ref.putClass(stringIDToTypeID("contentLayer"));
  desc.putReference(charIDToTypeID("null"), ref);

  var layerDesc = new ActionDescriptor();

  // Solid fill (we usually set fillOpacity=0; color is mostly irrelevant, but set it anyway).
  var fillDesc = new ActionDescriptor();
  var rgb = new ActionDescriptor();
  rgb.putDouble(charIDToTypeID("Rd  "), color.r);
  rgb.putDouble(charIDToTypeID("Grn "), color.g);
  rgb.putDouble(charIDToTypeID("Bl  "), color.b);
  fillDesc.putObject(charIDToTypeID("Clr "), charIDToTypeID("RGBC"), rgb);
  layerDesc.putObject(charIDToTypeID("Type"), stringIDToTypeID("solidColorLayer"), fillDesc);

  // Rectangle vector mask.
  var rect = new ActionDescriptor();
  rect.putUnitDouble(charIDToTypeID("Top "), charIDToTypeID("#Pxl"), top);
  rect.putUnitDouble(charIDToTypeID("Left"), charIDToTypeID("#Pxl"), left);
  rect.putUnitDouble(charIDToTypeID("Btom"), charIDToTypeID("#Pxl"), bottom);
  rect.putUnitDouble(charIDToTypeID("Rght"), charIDToTypeID("#Pxl"), right);
  layerDesc.putObject(charIDToTypeID("Shp "), charIDToTypeID("Rctn"), rect);

  desc.putObject(charIDToTypeID("Usng"), stringIDToTypeID("contentLayer"), layerDesc);
  executeAction(charIDToTypeID("Mk  "), desc, DialogModes.NO);
}

function _createBubbleRectangles() {
  var state = _hostState.createBubbleRectangles;
  if (!documents.length) {
    state.result = jamJSON.stringify({ error: "doc" });
    return;
  }

  var data = state.data || {};
  var groupName = data.groupName || "BUBBLES_DETECTED";
  var bubbles = data.bubbles || [];
  if (!bubbles.length) {
    state.result = jamJSON.stringify({ error: "bubbles" });
    return;
  }

  var stroke = data.stroke || { size: 4, opacity: 100, enabled: true, color: { r: 0, g: 120, b: 255 } };
  var fillOpacity = data.fillOpacity === undefined || data.fillOpacity === null ? 0 : data.fillOpacity;

  var startUnits = app.preferences.rulerUnits;
  try {
    app.preferences.rulerUnits = Units.PIXELS;
    var doc = app.activeDocument;
    var docW = 0;
    var docH = 0;
    try {
      docW = doc.width.value;
      docH = doc.height.value;
    } catch (eSize) { }

    var existing = _findLayerSetByName(doc, groupName);
    if (existing) {
      try {
        existing.remove();
      } catch (eRemove) { }
    }

    var group = doc.layerSets.add();
    group.name = groupName;

    // Convenience: ensure manual ROI group exists for xAI imagine step.
    // Do NOT delete if it already exists (user drawings must persist across re-detect).
    var imagineSquares = _findLayerSetByName(doc, "IMAGINE_SQUARES");
    if (!imagineSquares) {
      imagineSquares = doc.layerSets.add();
      imagineSquares.name = "IMAGINE_SQUARES";
      try {
        // Keep it above the bubbles group so it's easy to draw on top.
        imagineSquares.move(group, ElementPlacement.PLACEBEFORE);
      } catch (eMoveImagine) { }
    }

    for (var i = 0; i < bubbles.length; i++) {
      var b = bubbles[i] || {};
      var bb = b.bbox || {};
      var left = Math.max(0, Math.floor(bb.left || 0));
      var top = Math.max(0, Math.floor(bb.top || 0));
      if (docW > 0) left = Math.min(left, docW - 1);
      if (docH > 0) top = Math.min(top, docH - 1);

      var right = Math.max(left + 1, Math.ceil(bb.right || 0));
      var bottom = Math.max(top + 1, Math.ceil(bb.bottom || 0));
      if (docW > 0) right = Math.min(right, docW);
      if (docH > 0) bottom = Math.min(bottom, docH);

      if (right <= left) right = left + 1;
      if (bottom <= top) bottom = top + 1;

      _createRectShapeLayer(left, top, right, bottom, {
        r: stroke.color && stroke.color.r !== undefined ? stroke.color.r : 0,
        g: stroke.color && stroke.color.g !== undefined ? stroke.color.g : 120,
        b: stroke.color && stroke.color.b !== undefined ? stroke.color.b : 255
      });

      var layer = doc.activeLayer;
      layer.name = b.id || ("AUTO_" + _padNum(i + 1, 4));

      try {
        layer.fillOpacity = fillOpacity;
      } catch (eFill) { }

      _setLayerStroke(stroke);
      try {
        layer.move(group, ElementPlacement.INSIDE);
      } catch (eMove) { }
    }

    state.result = jamJSON.stringify({ ok: true, count: bubbles.length, groupName: groupName });
  } catch (e) {
    try {
      app.activeDocument.selection.deselect();
    } catch (eDese) { }
    state.result = jamJSON.stringify({ error: "create", detail: "" + e });
  } finally {
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function createBubbleRectangles(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var state = _hostState.createBubbleRectangles;
  state.data = data;
  state.result = "";
  app.activeDocument.suspendHistory("TypeR Auto Translate: Create Bubbles", "_createBubbleRectangles()");
  return state.result;
}

function getRectanglesFromGroup(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var groupName = (data && data.groupName) || "BUBBLES_DETECTED";
  var useNoEffects = !!(data && data.useNoEffects);
  if (groupName === "IMAGINE_SQUARES") useNoEffects = true;
  var startUnits = app.preferences.rulerUnits;
  try {
    app.preferences.rulerUnits = Units.PIXELS;
    var doc = app.activeDocument;
    var group = _findLayerSetByName(doc, groupName);
    if (!group) {
      return jamJSON.stringify({ error: "group" });
    }
    var layers = [];
    _collectArtLayers(group, layers);
    var out = [];
    for (var i = 0; i < layers.length; i++) {
      var layer = layers[i];
      var b = _getLayerBoundsPx(layer, useNoEffects);
      out.push({
        name: layer.name,
        left: b.left,
        top: b.top,
        right: b.right,
        bottom: b.bottom,
        width: b.width,
        height: b.height,
        xMid: b.xMid,
        yMid: b.yMid,
      });
    }
    return jamJSON.stringify({ rectangles: out });
  } catch (e) {
    return jamJSON.stringify({ error: "read", detail: "" + e });
  } finally {
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function _renumberBubbleGroup() {
  var state = _hostState.renumberBubbleGroup;
  if (!documents.length) {
    state.result = jamJSON.stringify({ error: "doc" });
    return;
  }

  var data = state.data || {};
  var groupName = data.groupName || "BUBBLES_DETECTED";
  var rowTolFactor = Number(data.rowToleranceFactor);
  if (!rowTolFactor || isNaN(rowTolFactor) || rowTolFactor <= 0) rowTolFactor = 0.5;

  var startUnits = app.preferences.rulerUnits;
  try {
    app.preferences.rulerUnits = Units.PIXELS;
    var doc = app.activeDocument;
    var group = _findLayerSetByName(doc, groupName);
    if (!group) {
      state.result = jamJSON.stringify({ error: "group" });
      return;
    }

    var layers = [];
    _collectArtLayers(group, layers);
    if (!layers.length) {
      state.result = jamJSON.stringify({ error: "empty" });
      return;
    }

    // NOTE: ExtendScript Array.sort is not guaranteed to be stable.
    // If the comparator returns 0 for different items, their relative order can become random.
    // We force deterministic ordering by always using a final tie-breaker.
    function _stableSort(arr, cmp) {
      var decorated = [];
      for (var iDec = 0; iDec < arr.length; iDec++) {
        decorated.push({ v: arr[iDec], i: iDec });
      }
      decorated.sort(function (a, b) {
        var c = cmp(a.v, b.v);
        if (c) return c;
        return a.i - b.i;
      });
      for (var iUnd = 0; iUnd < decorated.length; iUnd++) {
        arr[iUnd] = decorated[iUnd].v;
      }
    }

    // Horizontal reading direction (defaults to left→right).
    // Supported values: "ltr"/"rtl", "left-right"/"right-left", "left_to_right"/"right_to_left".
    var dir = (data.direction || data.readingOrder || data.order || "ltr");
    dir = ("" + dir).toLowerCase();
    var isRtl = dir === "rtl" || dir === "right-left" || dir === "rightleft" || dir === "right_to_left" || dir === "righttoleft" || dir === "rl";

    var items = [];
    for (var i = 0; i < layers.length; i++) {
      var layer = layers[i];
      var b = _getLayerBoundsPx(layer);
      items.push({ layer: layer, bounds: b, index: i });
    }

    // Sort by y first (top to bottom), then x (left to right / right to left).
    _stableSort(items, function (a, b) {
      var dy = a.bounds.yMid - b.bounds.yMid;
      if (dy) return dy;
      var ax = isRtl ? -a.bounds.xMid : a.bounds.xMid;
      var bx = isRtl ? -b.bounds.xMid : b.bounds.xMid;
      var dx = ax - bx;
      if (dx) return dx;
      var dt = a.bounds.top - b.bounds.top;
      if (dt) return dt;
      var al = isRtl ? -a.bounds.left : a.bounds.left;
      var bl = isRtl ? -b.bounds.left : b.bounds.left;
      var dl = al - bl;
      if (dl) return dl;
      return a.index - b.index;
    });

    // Group into rows by y proximity (not by bbox overlap).
    // Two items are considered to be on the same row if their yMid is within a tolerance
    // proportional to their heights (so tiny chat bubbles won't get merged into a single row).
    var rows = [];
    for (var j = 0; j < items.length; j++) {
      var item = items[j];
      var bestRow = null;
      var bestDist = null;
      for (var r = 0; r < rows.length; r++) {
        var row = rows[r];
        var tol = Math.max(1, Math.min(Math.abs(item.bounds.height), Math.abs(row.h)) * rowTolFactor);
        var dist = Math.abs(item.bounds.yMid - row.y);
        if (dist <= tol) {
          if (bestRow === null || dist < bestDist) {
            bestRow = row;
            bestDist = dist;
          }
        }
      }
      if (bestRow === null) {
        rows.push({
          y: item.bounds.yMid,
          h: Math.abs(item.bounds.height) || 0,
          minTop: item.bounds.top,
          items: [item],
        });
      } else {
        bestRow.items.push(item);
        var nRow = bestRow.items.length;
        bestRow.y = (bestRow.y * (nRow - 1) + item.bounds.yMid) / nRow;
        bestRow.h = (bestRow.h * (nRow - 1) + (Math.abs(item.bounds.height) || 0)) / nRow;
        bestRow.minTop = Math.min(bestRow.minTop, item.bounds.top);
      }
    }

    // Sort rows by their top-most item.
    _stableSort(rows, function (a, b) {
      var dt = a.minTop - b.minTop;
      if (dt) return dt;
      return a.y - b.y;
    });

    // Within each row, sort by x (respecting direction). Tie-break by y to keep vertical stacks deterministic.
    var ordered = [];
    for (var k = 0; k < rows.length; k++) {
      _stableSort(rows[k].items, function (a, b) {
        var ax = isRtl ? -a.bounds.xMid : a.bounds.xMid;
        var bx = isRtl ? -b.bounds.xMid : b.bounds.xMid;
        var dx = ax - bx;
        if (dx) return dx;
        var dy = a.bounds.yMid - b.bounds.yMid;
        if (dy) return dy;
        var dt = a.bounds.top - b.bounds.top;
        if (dt) return dt;
        return a.index - b.index;
      });
      for (var q = 0; q < rows[k].items.length; q++) {
        ordered.push(rows[k].items[q]);
      }
    }

    var pad = Math.max(3, ("" + ordered.length).length);
    var bubbles = [];
    for (var n = 0; n < ordered.length; n++) {
      var newId = "B" + _padNum(n + 1, pad);
      try {
        ordered[n].layer.name = newId;
      } catch (eName) { }
      var bb = ordered[n].bounds;
      bubbles.push({
        id: newId,
        bbox: { left: bb.left, top: bb.top, right: bb.right, bottom: bb.bottom },
        source: "edited",
        confidence: null,
      });
    }

    state.result = jamJSON.stringify({ bubbles: bubbles });
  } catch (e) {
    state.result = jamJSON.stringify({ error: "renumber", detail: "" + e });
  } finally {
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function renumberBubbleGroup(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var state = _hostState.renumberBubbleGroup;
  state.data = data;
  state.result = "";
  app.activeDocument.suspendHistory("TypeR Auto Translate: Renumber Bubbles", "_renumberBubbleGroup()");
  return state.result;
}

function _setActiveTextSizeAbs(size) {
  try {
    var ref = new ActionReference();
    ref.putProperty(charID.Property, charID.TextStyle);
    ref.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);
    var currentTextStyle = executeActionGet(ref);
    if (!currentTextStyle.hasKey(charID.TextStyle)) return false;
    var textStyle = currentTextStyle.getObjectValue(charID.TextStyle);
    var sizeUnit = textStyle.getUnitDoubleType(charID.Size);

    var descriptor = new ActionDescriptor();
    var reference = new ActionReference();
    reference.putProperty(charID.Property, charID.TextStyle);
    reference.putEnumerated(charID.TextLayer, charID.Ordinal, charID.Target);
    descriptor.putReference(charID.Null, reference);

    var newTextStyle = new ActionDescriptor();
    newTextStyle.putUnitDouble(charID.Size, sizeUnit, size);
    descriptor.putObject(charID.To, charID.TextStyle, newTextStyle);
    executeAction(charID.Set, descriptor, DialogModes.NO);
    return true;
  } catch (e) {
    return false;
  }
}

function _setActiveTextSize(size) {
  // Prefer DOM API first (more reliable across versions), fallback to ActionManager.
  try {
    if (app.activeDocument && app.activeDocument.activeLayer && app.activeDocument.activeLayer.textItem) {
      app.activeDocument.activeLayer.textItem.size = new UnitValue(Number(size), "pt");
      return true;
    }
  } catch (eDom) { }
  return _setActiveTextSizeAbs(size);
}



function _createTextLayerInBox(data, width, height) {
  var style = _ensureStyle(data.style);
  var text = data.text || "";
  // Photoshop uses \r for line breaks in text layers, not \n
  style.textProps.layerText.textKey = text.replace(/\n/g, "\r");
  style.textProps.layerText.textStyleRange[0].to = text.length;
  style.textProps.layerText.paragraphStyleRange[0].to = text.length;

  var sizeProp = style.textProps.layerText.textStyleRange[0].textStyle.size;
  if (typeof sizeProp !== "number") {
    try {
      var textParams = jamText.getLayerText();
      _hostState.fallbackTextSize = textParams.layerText.textStyleRange[0].textStyle.size;
    } catch (error) { }
    style.textProps.layerText.textStyleRange[0].textStyle.size = _hostState.fallbackTextSize;
  }

  style.textProps.layerText.textShape = [
    {
      textType: "box",
      orientation: "horizontal",
      bounds: {
        top: 0,
        left: 0,
        right: _convertPixelToPoint(width),
        bottom: _convertPixelToPoint(height),
      },
    },
  ];

  jamEngine.jsonPlay("make", {
    target: ["<reference>", [["textLayer", ["<class>", null]]]],
    using: jamText.toLayerTextObject(style.textProps),
  });
  _applyMiddleEast(style.textProps.layerText.textStyleRange[0].textStyle);
  if (style.stroke) {
    _setLayerStroke(style.stroke);
  }
  if (data.direction) {
    _applyTextDirection(data.direction, text.length);
  }
}

function _createTranslatedTextLayers() {
  var state = _hostState.createTranslatedTextLayers;
  if (!documents.length) {
    state.result = jamJSON.stringify({ error: "doc" });
    return;
  }

  var data = state.data || {};
  var groupName = data.groupName || "TRANSLATION";
  var bubbles = data.bubbles || [];
  var translations = data.translations || [];
  if (!bubbles.length) {
    state.result = jamJSON.stringify({ error: "bubbles" });
    return;
  }

  var tMap = {};
  for (var i = 0; i < translations.length; i++) {
    var t = translations[i] || {};
    if (t.id) tMap[t.id] = t.text || "";
  }

  var fit = data.fit || {};
  var padding = fit.padding || 0;
  var fitMap = fit.map || fit.byId || fit.items || null;
  var styles = data.styles || {};
  var baseStyle = _ensureStyle(data.style);
  baseStyle = _applyAutoTranslateDefaults(baseStyle);
  var append = !!data.append;

  var startUnits = app.preferences.rulerUnits;
  try {
    app.preferences.rulerUnits = Units.PIXELS;
    var doc = app.activeDocument;
    var docW = 0;
    var docH = 0;
    try {
      docW = doc.width.value;
      docH = doc.height.value;
    } catch (eSize) { }

    var existing = _findLayerSetByName(doc, groupName);
    if (existing && !append) {
      try {
        existing.remove();
      } catch (eRemove) { }
    }
    var group = existing && append ? existing : doc.layerSets.add();
    try {
      group.name = groupName;
    } catch (eGroupName) { }

  var created = 0;
    for (var j = 0; j < bubbles.length; j++) {
      var b = bubbles[j] || {};
      var id = b.id;
      var bb = b.bbox || {};
      var entry = (fitMap && id && fitMap[id]) ? fitMap[id] : null;
      var bbFit = entry && entry.layout_bbox ? entry.layout_bbox : null;
      var box = (bb && bb.left != null && bb.right != null && bb.top != null && bb.bottom != null) ? bb : (bbFit || bb);
      var left = Math.max(0, Math.floor(box.left || 0));
      var top = Math.max(0, Math.floor(box.top || 0));
      if (docW > 0) left = Math.min(left, docW - 1);
      if (docH > 0) top = Math.min(top, docH - 1);

      var right = Math.max(left + 1, Math.ceil(box.right || 0));
      var bottom = Math.max(top + 1, Math.ceil(box.bottom || 0));
      if (docW > 0) right = Math.min(right, docW);
      if (docH > 0) bottom = Math.min(bottom, docH);

      if (right <= left) right = left + 1;
      if (bottom <= top) bottom = top + 1;
      var boxW = Math.max(1, right - left);
      var boxH = Math.max(1, bottom - top);
      var text = tMap[id] || "";

      var innerW = Math.max(1, boxW - padding * 2);
      var innerH = Math.max(1, boxH - padding * 2);

      var overrides = (styles && id && styles[id]) ? styles[id] : null;
      var target = {
        xMid: left + boxW / 2,
        yMid: top + boxH / 2,
      };

      var bubbleStyle = _applyStyleOverrides(baseStyle, overrides);
      _createTextLayerInBox({ text: text, style: bubbleStyle, direction: data.direction }, innerW, innerH);
      try {
        doc.activeLayer.name = id;
      } catch (eName) { }

      if (text && text.length) {
        var pt = entry && entry.font_pt != null ? Number(entry.font_pt) : 18;
        if (!(pt > 0)) pt = 18;
        _setActiveTextSize(pt);
      }
      var bounds = _getLayerBoundsPx(doc.activeLayer);
      _positionLayerWithinSelection(target, bounds);

      try {
        doc.activeLayer.move(group, ElementPlacement.INSIDE);
      } catch (eMove) { }
      created++;
    }

    state.result = jamJSON.stringify({ ok: true, count: created, groupName: groupName });
  } catch (e) {
    state.result = jamJSON.stringify({ error: "create", detail: "" + e });
  } finally {
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function createTranslatedTextLayers(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var state = _hostState.createTranslatedTextLayers;
  state.data = data;
  state.result = "";
  app.activeDocument.suspendHistory("TypeR Auto Translate: Apply Translation", "_createTranslatedTextLayers()");
  return state.result;
}

function _applyCleanedPng() {
  var state = _hostState.applyCleanedPng;
  if (!documents.length) {
    state.result = jamJSON.stringify({ error: "doc" });
    return;
  }

  var data = state.data || {};
  var cleanedPath = data.path || "";
  if (!cleanedPath) {
    state.result = jamJSON.stringify({ error: "path" });
    return;
  }

  var groupName = data.groupName || "CLEANED";
  var belowGroupName = data.belowGroupName || "BUBBLES_DETECTED";
  var translationGroupName = data.translationGroupName || "TRANSLATION";

  var startUnits = app.preferences.rulerUnits;
  var tmpDoc = null;
  try {
    app.preferences.rulerUnits = Units.PIXELS;

    var targetDoc = app.activeDocument;

    var existing = _findLayerSetByName(targetDoc, groupName);
    if (existing) {
      try {
        existing.remove();
      } catch (eRemove) { }
    }

    var group = targetDoc.layerSets.add();
    group.name = groupName;

    var file = new File(cleanedPath);
    if (!file.exists) {
      state.result = jamJSON.stringify({ error: "file_missing", detail: cleanedPath });
      return;
    }

    tmpDoc = app.open(file);
    try {
      tmpDoc.changeMode(ChangeMode.RGB);
    } catch (eMode) { }
    try {
      tmpDoc.flatten();
    } catch (eFlat) { }

    var layerToDup = tmpDoc.activeLayer;
    var dupLayer = null;
    try {
      dupLayer = layerToDup.duplicate(targetDoc, ElementPlacement.PLACEATBEGINNING);
    } catch (eDup) {
      dupLayer = layerToDup.duplicate(targetDoc);
    }

    try {
      tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
    } catch (eClose) { }
    tmpDoc = null;

    app.activeDocument = targetDoc;
    try {
      dupLayer.name = "CLEANED_IMAGE";
    } catch (eName) { }
    try {
      dupLayer.move(group, ElementPlacement.INSIDE);
    } catch (eMove) { }

    // Ensure CLEANED stays under BUBBLES/TRANSLATION so rectangles/text stay visible.
    var below = _findLayerSetByName(targetDoc, belowGroupName);
    if (!below) below = _findLayerSetByName(targetDoc, translationGroupName);
    if (below) {
      _moveLayerBelow(targetDoc, group, below);
    }

    state.result = jamJSON.stringify({ ok: true, groupName: groupName });
  } catch (e) {
    state.result = jamJSON.stringify({ error: "apply_cleaned", detail: "" + e });
  } finally {
    try {
      if (tmpDoc) tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
    } catch (eClose2) { }
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function applyCleanedPng(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var state = _hostState.applyCleanedPng;
  state.data = data;
  state.result = "";
  app.activeDocument.suspendHistory("TypeR Auto Translate: Apply Cleaned Image", "_applyCleanedPng()");
  return state.result;
}

function _findArtLayerByName(parent, name) {
  if (!parent || !name) return null;
  var layers = parent.artLayers || [];
  for (var i = 0; i < layers.length; i++) {
    var layer = layers[i];
    if (layer && layer.name === name) return layer;
  }
  var sets = parent.layerSets || [];
  for (var j = 0; j < sets.length; j++) {
    var inner = _findArtLayerByName(sets[j], name);
    if (inner) return inner;
  }
  return null;
}

function _applyImaginePatches() {
  var state = _hostState.applyImaginePatches;
  if (!documents.length) {
    state.result = jamJSON.stringify({ error: "doc" });
    return;
  }

  var data = state.data || {};
  var patches = data.patches || [];
  if (!patches.length) {
    state.result = jamJSON.stringify({ error: "patches" });
    return;
  }

  var groupName = data.groupName || "IMAGINE_PATCHES";
  var cleanedGroupName = data.cleanedGroupName || "CLEANED";
  var cleanedImageLayerName = data.cleanedImageLayerName || "CLEANED_IMAGE";
  var belowGroupName = data.belowGroupName || "BUBBLES_DETECTED";
  var translationGroupName = data.translationGroupName || "TRANSLATION";
  var replaceExisting = data.replaceExisting !== false;

  var startUnits = app.preferences.rulerUnits;
  var tmpDoc = null;
  try {
    app.preferences.rulerUnits = Units.PIXELS;

    var targetDoc = app.activeDocument;
    var cleanedGroup = _findLayerSetByName(targetDoc, cleanedGroupName);

    // Remove existing patches group (either under CLEANED or at top-level).
    if (replaceExisting) {
      var existing = cleanedGroup ? _findLayerSetByName(cleanedGroup, groupName) : _findLayerSetByName(targetDoc, groupName);
      if (existing) {
        try {
          existing.remove();
        } catch (eRemove) { }
      }
    }

    var group = null;
    if (cleanedGroup) {
      group = cleanedGroup.layerSets.add();
      group.name = groupName;
      // Ensure it's above the CLEANED_IMAGE layer when present.
      var cleanedLayer = _findArtLayerByName(cleanedGroup, cleanedImageLayerName);
      if (cleanedLayer) {
        try {
          group.move(cleanedLayer, ElementPlacement.PLACEBEFORE);
        } catch (eMoveGroup) { }
      }
    } else {
      group = targetDoc.layerSets.add();
      group.name = groupName;
      // Place below BUBBLES/TRANSLATION so overlays remain visible.
      var below = _findLayerSetByName(targetDoc, belowGroupName);
      if (!below) below = _findLayerSetByName(targetDoc, translationGroupName);
      if (below) {
        _moveLayerBelow(targetDoc, group, below);
      }
    }

    var applied = 0;
    for (var i = 0; i < patches.length; i++) {
      var p = patches[i] || {};
      var filePath = p.path || p.file || "";
      var left = Number(p.left);
      var top = Number(p.top);
      if (!filePath || isNaN(left) || isNaN(top)) continue;

      var file = new File(filePath);
      if (!file.exists) continue;

      tmpDoc = app.open(file);
      try {
        tmpDoc.changeMode(ChangeMode.RGB);
      } catch (eMode) { }
      try {
        tmpDoc.flatten();
      } catch (eFlat) { }

      var layerToDup = tmpDoc.activeLayer;
      var dupLayer = null;
      try {
        dupLayer = layerToDup.duplicate(targetDoc, ElementPlacement.PLACEATBEGINNING);
      } catch (eDup) {
        dupLayer = layerToDup.duplicate(targetDoc);
      }

      try {
        tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
      } catch (eClose) { }
      tmpDoc = null;

      app.activeDocument = targetDoc;
      try {
        dupLayer.name = p.id || ("PATCH_" + _padNum(i + 1, 4));
      } catch (eName) { }

      // Position by bbox top-left (server already resized the patch to ROI size).
      try {
        var b = _getLayerBoundsPx(dupLayer);
        dupLayer.translate(left - b.left, top - b.top);
      } catch (ePos) { }

      try {
        dupLayer.move(group, ElementPlacement.INSIDE);
      } catch (eMove) { }
      applied++;
    }

    state.result = jamJSON.stringify({ ok: true, count: applied, groupName: groupName });
  } catch (e) {
    state.result = jamJSON.stringify({ error: "apply_imagine", detail: "" + e });
  } finally {
    try {
      if (tmpDoc) tmpDoc.close(SaveOptions.DONOTSAVECHANGES);
    } catch (eClose2) { }
    try {
      app.preferences.rulerUnits = startUnits;
    } catch (e2) { }
  }
}

function applyImaginePatches(data) {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var state = _hostState.applyImaginePatches;
  state.data = data;
  state.result = "";
  app.activeDocument.suspendHistory("TypeR Auto Translate: Apply Imagine Patches", "_applyImaginePatches()");
  return state.result;
}

function setActiveLayerText(data) {
  var state = _hostState.setActiveLayerText;
  state.data = data;
  state.result = "";
  app.activeDocument.suspendHistory("TyperTools Change", "_setActiveLayerText()");
  return state.result;
}

function createTextLayerInSelection(data, point) {
  var state = _hostState.createTextLayerInSelection;
  state.data = data;
  state.point = point;
  state.padding = data.padding || 0;
  state.result = "";
  app.activeDocument.suspendHistory("TyperTools Paste", "_createTextLayerInSelection()");
  return state.result;
}

function alignTextLayerToSelection(data) {
  var state = _hostState.alignTextLayerToSelection;
  state.resize = !!data.resizeTextBox;
  state.padding = data.padding || 0;
  state.result = "";
  app.activeDocument.suspendHistory("TyperTools Align", "_alignTextLayerToSelection()");
  return state.result;
}

function alignTextLayerToTarget(data) {
  var state = _hostState.alignTextLayerToTarget;
  state.resize = !!data.resizeTextBox;
  state.padding = data.padding || 0;
  state.target = data.target || null;
  state.result = "";
  app.activeDocument.suspendHistory("TyperTools Align (AI)", "_alignTextLayerToTarget()");
  return state.result;
}

function changeActiveLayerTextSize(val) {
  var state = _hostState.changeActiveLayerTextSize;
  state.value = val;
  state.result = "";
  app.activeDocument.suspendHistory("TyperTools Resize", "_changeActiveLayerTextSize()");
  return state.result;
}

function getCurrentSelection() {
  if (!documents.length) {
    return jamJSON.stringify({ error: "doc" });
  }
  var selection = _checkSelection({ adjustAmount: 0 });
  if (selection.error) {
    return jamJSON.stringify({ error: selection.error });
  }
  return jamJSON.stringify(selection);
}

function startSelectionMonitoring() {
  var monitor = _hostState.selectionMonitor;
  // Démarrer la surveillance des changements de sélection
  try {
    if (monitor.callback) {
      try {
        app.removeNotifier("Slct", monitor.callback);
      } catch (eRemove) { }
    }
  } catch (eRemove2) { }

  monitor.pendingSelection = null;
  monitor.pendingBoundsKey = null;
  monitor.lastPollMs = 0;

  monitor.callback = function () {
    try {
      if (!documents.length) return;
      var currentSelection = _checkSelection({ adjustAmount: 0 });
      if (!currentSelection.error) {
        var currentBounds = _selectionBoundsKey(currentSelection);
        if (currentBounds !== monitor.lastBoundsKey) {
          monitor.lastBoundsKey = currentBounds;
          monitor.pendingSelection = currentSelection;
          monitor.pendingBoundsKey = currentBounds;
          // Notifier l'extension CEP du changement (Mac only workaround)
          if ($.os.toLowerCase().indexOf("mac") !== -1) {
            try {
              app.system("osascript -e 'tell application \"System Events\" to keystroke \"x\" using {command down, option down, shift down}'");
            } catch (eSystem) { }
          }
        }
      }
    } catch (e) {
      // ignore
    }
  };

  try {
    app.addNotifier("Slct", monitor.callback);
  } catch (eAdd) {
    monitor.callback = null;
  }
}

function stopSelectionMonitoring() {
  var monitor = _hostState.selectionMonitor;
  if (monitor.callback) {
    try {
      app.removeNotifier("Slct", monitor.callback);
    } catch (eRemove) { }
    monitor.callback = null;
  }
  monitor.lastBoundsKey = null;
  monitor.pendingSelection = null;
  monitor.pendingBoundsKey = null;
  monitor.lastPollMs = 0;
}

function getSelectionChanged() {
  var monitor = _hostState.selectionMonitor;
  var shiftPressed = false;
  try {
    var keyboardState = ScriptUI.environment && ScriptUI.environment.keyboardState;
    shiftPressed = !!(keyboardState && keyboardState.shiftKey);
  } catch (eKeyboard) { }

  if (monitor.pendingSelection && !monitor.pendingSelection.error) {
    var pending = monitor.pendingSelection;
    monitor.pendingSelection = null;
    monitor.pendingBoundsKey = null;
    return jamJSON.stringify({
      shiftKey: shiftPressed,
      top: pending.top,
      left: pending.left,
      right: pending.right,
      bottom: pending.bottom,
      width: pending.width,
      height: pending.height,
      xMid: pending.xMid,
      yMid: pending.yMid,
    });
  }

  // Fallback polling mode (when notifier is not available / failed to register)
  if (!documents.length) {
    monitor.lastBoundsKey = null;
    return jamJSON.stringify({ noChange: true, shiftKey: shiftPressed });
  }

  // If notifier is available, avoid constant polling; keep a slow fallback poll for robustness.
  if (monitor.callback) {
    var nowMs = 0;
    try {
      nowMs = new Date().getTime();
    } catch (eTime) { }
    if (nowMs && monitor.lastPollMs && nowMs - monitor.lastPollMs < 1000) {
      return jamJSON.stringify({ noChange: true, shiftKey: shiftPressed });
    }
    monitor.lastPollMs = nowMs || 0;
  }

  var currentSelection;
  try {
    currentSelection = _checkSelection({ adjustAmount: 0 });
  } catch (eSel) {
    currentSelection = { error: "selection" };
  }

  if (currentSelection && !currentSelection.error) {
    var currentBounds = _selectionBoundsKey(currentSelection);
    if (currentBounds !== monitor.lastBoundsKey) {
      monitor.lastBoundsKey = currentBounds;
      return jamJSON.stringify({
        shiftKey: shiftPressed,
        top: currentSelection.top,
        left: currentSelection.left,
        right: currentSelection.right,
        bottom: currentSelection.bottom,
        width: currentSelection.width,
        height: currentSelection.height,
        xMid: currentSelection.xMid,
        yMid: currentSelection.yMid,
      });
    }
  }
  return jamJSON.stringify({ noChange: true, shiftKey: shiftPressed });
}

function _createTextLayersInStoredSelections() {
  var state = _hostState.createTextLayersInStoredSelections;
  if (!documents.length) {
    state.result = "doc";
    return;
  }

  var texts = state.data.texts || [];
  var styles = state.data.styles || [];

  if (texts.length === 0 || state.selections.length === 0) {
    state.result = "noSelection";
    return;
  }

  var maxCount = Math.min(texts.length, state.selections.length);

  for (var i = 0; i < maxCount; i++) {
    var text = texts[i] || texts[texts.length - 1] || "";
    var baseStyle = styles[i] || styles[styles.length - 1] || null;
    var style = _ensureStyle(baseStyle);
    var selection = state.selections[i];

    if (!text) continue;

    var dimensions = _calculateSelectionDimensions(selection, state.padding);

    // Créer le layer de texte
    var data = { text: text, style: style, direction: state.data.direction };
    _createAndSetLayerText(data, dimensions.width, dimensions.height);

    var bounds = _getCurrentTextLayerBounds();
    if (state.point) {
      _changeToPointText();
    } else {
      _resizeTextBoxToContent(dimensions.width, bounds);
    }
    bounds = _getCurrentTextLayerBounds();

    // Positionner le layer à l'emplacement de la sélection stockée
    _positionLayerWithinSelection(selection, bounds);
  }

  // Vider les sélections stockées après utilisation
  state.selections = [];
  state.result = "";
}

function createTextLayersInStoredSelections(data, point) {
  var state = _hostState.createTextLayersInStoredSelections;
  state.data = data;
  state.point = point;
  state.padding = data.padding || 0;
  state.result = "";

  // Les sélections sont passées directement depuis React
  if (data && data.selections) {
    state.selections = data.selections;
  } else {
    state.selections = [];
  }

  app.activeDocument.suspendHistory("TyperTools Multiple Paste", "_createTextLayersInStoredSelections()");
  return state.result;
}

function openFile(path, autoClose) {
  if (autoClose && _hostState.lastOpenedDocId !== null) {
    for (var i = 0; i < app.documents.length; i++) {
      var doc = app.documents[i];
      if (doc.id === _hostState.lastOpenedDocId) {
        try {
          doc.close(SaveOptions.SAVECHANGES);
        } catch (e) { }
        break;
      }
    }
  }
  var newDoc = app.open(File(path));
  if (autoClose) {
    _hostState.lastOpenedDocId = newDoc.id;
  }
}

function deleteFolder(folderPath) {
  try {
    var folder = new Folder(folderPath);
    if (folder.exists) {
      // Recursively delete contents
      var files = folder.getFiles();
      for (var i = 0; i < files.length; i++) {
        if (files[i] instanceof Folder) {
          deleteFolder(files[i].fsName);
        } else {
          files[i].remove();
        }
      }
      folder.remove();
    }
    return 'OK';
  } catch (e) {
    return 'ERROR: ' + e.message;
  }
}

function openFolder(folderPath) {
  try {
    var os = $.os.toLowerCase();
    if (os.indexOf('win') !== -1) {
      // Windows: open Explorer
      app.system('explorer "' + folderPath.replace(/\//g, '\\') + '"');
    } else {
      // macOS: open Finder
      app.system('open "' + folderPath + '"');
    }
    return 'OK';
  } catch (e) {
    return 'ERROR: ' + e.message;
  }
}

function makeExecutable(filePath) {
  try {
    var os = $.os.toLowerCase();
    if (os.indexOf('mac') !== -1) {
      app.system('chmod +x "' + filePath + '"');
    }
    return 'OK';
  } catch (e) {
    return 'ERROR: ' + e.message;
  }
}
