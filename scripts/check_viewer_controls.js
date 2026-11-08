/* Run the real viewer application against a small DOM/3Dmol adapter.
 * This exercises event wiring and saved-coordinate handling, not WebGL rendering.
 * Usage: node scripts/check_viewer_controls.js
 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(root, "viewer_assets/viewer.js"), "utf8");
const template = fs.readFileSync(path.join(root, "viewer_assets/viewer.html"), "utf8");

class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.attributes = {};
    this.style = {}; this._text = ""; this.value = ""; this.checked = false;
    this.disabled = false; this.hidden = false; this.className = "";
    this.classList = {toggle: () => {}};
  }
  set textContent(text) { this._text = String(text); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  get options() { return this.children; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = children; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  click() { this.clicks = (this.clicks || 0) + 1; return this.onclick?.(); }
}

function fixture() {
  const frame = (name, coords) => ({molblock: name, coords, xyz: "3\nsaved coordinates\n"});
  const first = {conformer_index: 0, geometry_screen_passed: true, refinement_status: "converged",
    final: frame("saved-final", [[0.123456789, 0, 0], [1.623456789, 0, 0], [0, 1, 0]]),
    raw: frame("saved-raw", [[0, 0.321098765, 0], [2, 0, 0], [0, 1, 0]])};
  const second = {conformer_index: 1, geometry_screen_passed: false,
    final: frame("saved-second", [[0, 0, 1.987654321], [1, 0, 0], [0, 1, 0]])};
  return {source: "runtime-fixture", candidates: [
    {graph_index: 0, formula: "HNO", smiles: "synthetic adapter fixture", elements: ["N", "O", "H"], svg: "<svg/>", conformers: [first, second]},
    {graph_index: 1, formula: "CF", smiles: "synthetic adapter fixture", elements: ["C", "F"], svg: "<svg/>", conformers: []}
  ], input_ir: {frequencies: [3000, 1500], intensities: [20, 10]}};
}

function boot(data, failWebGL = false) {
  const elements = new Map([...template.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const get = id => { assert.ok(elements.has(id), `viewer accesses an unknown HTML id: ${id}`); return elements.get(id); };
  get("prediction-data").textContent = JSON.stringify(data);
  get("geometry").append(new Element("option"), new Element("option"));
  get("geometry").value = "final"; get("style").value = "ball";
  get("hydrogens").checked = true;
  const events = {}, calls = [], models = [], downloads = [];
  let clickable;
  const frames = new Map();
  data.candidates.forEach(c => c.conformers.forEach(sample => [sample.final, sample.raw].filter(Boolean).forEach(frame => frames.set(frame.molblock, {frame, elements: c.elements}))));
  const record = method => (...args) => { calls.push([method, ...args]); };
  const viewer = {};
  for (const method of ["setProjection", "removeAllModels", "removeAllShapes", "removeAllLabels", "setStyle", "render", "spin", "addLine", "setView", "zoomTo", "rotate", "zoom", "translate", "resize"]) viewer[method] = record(method);
  viewer.getView = () => [1, 2, 3, 4];
  viewer.addLabel = () => { throw new Error("Atom labels must remain outside the 3D viewport."); };
  viewer.pngURI = () => "data:image/png;base64,fixture";
  viewer.addModel = (molblock, format, options) => {
    calls.push(["addModel", molblock, format, options]);
    const match = frames.get(molblock); assert.ok(match, "only saved molblocks may be rendered");
    const atoms = match.frame.coords.map((_, index) => ({index, elem: match.elements[index], x: -9, y: -9, z: -9}));
    const model = {atoms, selectedAtoms: () => atoms}; models.push(model); return model;
  };
  viewer.setClickable = (selection, enabled, callback) => { assert.equal(enabled, true); clickable = callback; };
  const document = {
    getElementById: get,
    createElement: tag => { const element = new Element(tag); if (tag === "a") element.click = () => downloads.push({href: element.href, filename: element.download}); return element; },
    createElementNS: (_ns, tag) => new Element(tag),
    createTextNode: text => { const element = new Element("text"); element.textContent = text; return element; },
    querySelectorAll: selector => { assert.equal(selector, ".candidate"); return get("candidates").children; },
    addEventListener: (name, handler) => { events[name] = handler; },
    fullscreenElement: null,
    exitFullscreen: async () => { document.fullscreenElement = null; events.fullscreenchange(); },
    hidden: false
  };
  get("scene-card").requestFullscreen = async () => { document.fullscreenElement = get("scene-card"); events.fullscreenchange(); };
  const context = {
    document, Blob, URL: {createObjectURL: () => "blob:fixture", revokeObjectURL: record("revokeObjectURL")},
    setTimeout: callback => callback(),
    ResizeObserver: class { constructor(callback) { this.callback = callback; } observe() { this.callback(); } },
    $3Dmol: {createViewer: () => { if (failWebGL) throw new Error("WebGL unavailable"); return viewer; }}
  };
  vm.runInNewContext(source, context, {filename: "viewer_assets/viewer.js"});
  return {get, viewer, calls, models, downloads, document, events, atomClick: index => clickable(models.at(-1).atoms[index])};
}

async function main() {
  const data = fixture(), app = boot(data), {get, calls, models} = app;
  assert.equal(get("element-legend").textContent, "Nitrogen (N)Oxygen (O)Hydrogen (H)", "legend must use candidate.elements, never infer from formula or SMILES");
  assert.equal(models[0].atoms[0].x, data.candidates[0].conformers[0].final.coords[0][0], "restore full saved coordinate precision after parsing the rounded molfile");
  app.atomClick(0); assert.match(get("picked").textContent, /0\.123/);
  get("measure").click(); app.atomClick(0); app.atomClick(1);
  assert.match(get("picked").textContent, /Separation: 1\.500/, "distance belongs in the external inspector");
  assert.ok(calls.some(call => call[0] === "addLine"), "measurement can draw a line without viewport labels");
  get("left").click(); get("right").click(); get("tilt").click();
  for (const expected of [["rotate", -20, "y"], ["rotate", 20, "y"], ["rotate", 20, "x"]]) assert.ok(calls.some(call => JSON.stringify(call) === JSON.stringify(expected)));
  get("zoom-in").click(); get("zoom-out").click();
  assert.ok(calls.some(call => call[0] === "zoom" && call[1] === 1.2));
  assert.ok(calls.some(call => call[0] === "zoom" && call[1] === 1 / 1.2));
  for (const id of ["pan-left", "pan-right", "pan-up", "pan-down"]) get(id).click();
  assert.equal(calls.filter(call => call[0] === "translate").length, 4);
  get("spin").click(); assert.equal(get("spin").attributes["aria-pressed"], "true");
  app.document.hidden = true; app.events.visibilitychange(); assert.equal(get("spin").attributes["aria-pressed"], "false");
  get("spin").click(); get("reset").click(); assert.equal(get("spin").attributes["aria-pressed"], "false");
  get("geometry").value = "raw"; get("geometry").onchange();
  assert.equal(models.at(-1).atoms[0].y, 0.321098765);
  assert.ok(calls.some(call => call[0] === "setView"), "switching coordinates preserves the camera");
  get("conformer").value = "1"; get("geometry").value = "final"; get("conformer").onchange();
  assert.equal(models.at(-1).atoms[0].z, 1.987654321);
  get("style").value = "space"; get("style").onchange();
  assert.ok(calls.some(call => call[0] === "setStyle" && call[2].sphere?.scale === .92));
  get("hydrogens").checked = false; get("hydrogens").onchange(); assert.match(get("element-legend").textContent, /Hydrogen \(H\).*hidden/);
  assert.ok(calls.some(call => call[0] === "setStyle" && call[1].elem === "H" && call[2].sphere.hidden));
  await get("fullscreen").click(); assert.equal(get("fullscreen").textContent, "Exit full screen");
  await get("fullscreen").click(); assert.equal(get("fullscreen").textContent, "Full screen");
  get("xyz").click(); get("png").click(); assert.equal(app.downloads.length, 2);
  assert.match(app.downloads[0].filename, /candidate1_conformer2_final\.xyz$/);
  assert.match(app.downloads[1].href, /^data:image\/png/);
  get("candidates").children[1].click();
  assert.equal(get("element-legend").textContent, "Carbon (C)Fluorine (F)");
  assert.equal(get("spin").disabled, true); assert.equal(get("empty").style.display, "grid");

  const generated = fixture(); generated.provenance = {kind: "generated_from_2d"}; generated.input_ir = null;
  const from2d = boot(generated);
  assert.equal(from2d.get("ir-card").hidden, true, "2D-only input must not invent an IR panel");
  assert.match(from2d.get("coordinate-note").textContent, /not a measured or model-predicted/);
  assert.match(from2d.get("footer-note").textContent, /not an IR prediction/);
  const failed = boot(fixture(), true);
  assert.match(failed.get("empty").textContent, /Enable WebGL/); assert.equal(failed.get("spin").disabled, true);
  const empty = fixture(); empty.candidates = [];
  const noCandidate = boot(empty); assert.match(noCandidate.get("empty").textContent, /No replacement structure/);
  console.log("Viewer runtime checks passed: saved coordinates, external legend/inspection, all camera controls, exports, generated-structure mode, and failure states.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
