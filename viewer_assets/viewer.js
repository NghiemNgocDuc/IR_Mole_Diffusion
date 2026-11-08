"use strict";
(() => {
  const data = JSON.parse(document.getElementById("prediction-data").textContent);
  const $ = id => document.getElementById(id);
  const colors = {C: "#475569", N: "#3055bf", O: "#df3e4f", H: "#e9eef1", F: "#268858"};
  const elementNames = {C: "Carbon", N: "Nitrogen", O: "Oxygen", H: "Hydrogen", F: "Fluorine"};
  const generated = data.provenance?.kind === "generated_from_2d";
  let viewer, currentModel, graphPosition = 0, conformerPosition = 0, spinning = false;
  let measuring = false, firstAtom = null;
  const candidate = () => data.candidates[graphPosition];
  const conformer = () => candidate()?.conformers[conformerPosition];
  const frame = () => conformer()?.[$("geometry").value];
  $("source").textContent = data.source + " · " + (generated ? "3D conformers generated from a 2D structure" : data.candidates.length + " saved graph candidates");
  $("coordinate-note").textContent = generated ?
    "Generated from the supplied molecular graph. A plausible conformer, not a measured or model-predicted structure." :
    "Actual saved model coordinates. Rotation changes the camera, not the molecule.";
  if (generated) {
    $("candidate-heading").textContent = "Supplied structure";
    $("candidate-note").textContent = "3D conformers generated from this graph.";
    $("candidate-disclaimer").textContent = "A 2D structure can have multiple 3D conformations. These are generated estimates; they do not establish IR agreement.";
    $("footer-note").textContent = "Generated conformers from the supplied molecular graph; not an IR prediction or experimental geometry. No forward IR spectrum is calculated.";
    $("geometry").options[0].textContent = "Generated final";
  }

  function announce(message) { $("picked").textContent = message; }
  function clearMeasurement() {
    firstAtom = null;
    viewer?.removeAllShapes();
    refreshLabels();
    viewer?.render();
    announce(measuring ? "Click two atoms to measure their separation." : "Click an atom to inspect its saved coordinates.");
  }
  function refreshLabels() {
    if (!viewer) return;
    // Keep the viewport free of atom names and text, including measurements.
    viewer.removeAllLabels();
  }
  function elementLegend() {
    const legend = $("element-legend"); legend.replaceChildren();
    const present = new Set(candidate()?.elements || []);
    for (const symbol of ["C", "N", "O", "H", "F"]) {
      if (!present.has(symbol)) continue;
      const item = document.createElement("span"), dot = document.createElement("i");
      dot.className = "dot"; dot.style.backgroundColor = colors[symbol]; dot.setAttribute("aria-hidden", "true");
      item.append(dot, document.createTextNode(elementNames[symbol] + " (" + symbol + ")"));
      if (symbol === "H" && !$("hydrogens").checked) {
        item.className = "element-hidden"; item.append(document.createTextNode(" · hidden"));
      }
      legend.append(item);
    }
  }
  function applyStyle() {
    if (!viewer || !currentModel) return;
    const colorfunc = atom => colors[atom.elem] || "#8b5ca8";
    const mode = $("style").value;
    const style = mode === "space" ? {sphere: {scale: .92, colorfunc}} :
      mode === "stick" ? {stick: {radius: .17, colorfunc}} :
      {stick: {radius: .12, colorfunc}, sphere: {scale: .27, colorfunc}};
    viewer.setStyle({}, style);
    if (!$("hydrogens").checked) viewer.setStyle({elem: "H"}, {sphere: {hidden: true}, stick: {hidden: true}});
    refreshLabels(); elementLegend(); viewer.render();
  }
  function addMetric(name, value) {
    const term = document.createElement("dt"), detail = document.createElement("dd");
    term.textContent = name; detail.textContent = value; $("metrics").append(term, detail);
  }
  function details() {
    const c = conformer(); $("metrics").replaceChildren();
    const raw = $("geometry").value === "raw";
    $("status").className = "status" + (!c || raw || c.geometry_screen_passed !== true ? " warning" : "");
    $("status").textContent = generated ? "Generated conformer · not an experimental or IR-predicted geometry." : !c ? "No coordinate sample was saved." : raw ?
      "Raw sample · geometry screen below applies to the final saved sample." :
      "Final geometry screen: " + (c.geometry_screen_passed === true ? "PASS" : c.geometry_screen_passed === false ? "FAIL" : "unavailable");
    if (!c) return;
    addMetric("Saved refinement status", c.refinement_status || "unavailable");
    addMetric("Final bond penalty", Number.isFinite(c.bond_penalty) ? c.bond_penalty.toFixed(4) : "unavailable");
    addMetric("Final clash penalty", Number.isFinite(c.clash_penalty) ? c.clash_penalty.toFixed(4) : "unavailable");
    addMetric("Final MMFF energy (kcal/mol)", Number.isFinite(c.mmff_energy_kcal_mol) ? c.mmff_energy_kcal_mol.toFixed(3) : "unavailable");
    addMetric("Coordinates displayed", generated ? (raw ? "Embedded before refinement" : "Generated final") : raw ? "Raw sampled" : "Final saved");
  }
  function atomClicked(atom) {
    const coords = frame()?.coords[atom.index];
    if (!coords) return;
    if (!measuring) {
      announce(`${atom.elem}${atom.index + 1}: x ${coords[0].toFixed(3)}, y ${coords[1].toFixed(3)}, z ${coords[2].toFixed(3)} Å`);
      return;
    }
    if (firstAtom === null) {
      firstAtom = atom.index; announce(`Selected ${atom.elem}${atom.index + 1}. Click a second atom.`); return;
    }
    const first = frame().coords[firstAtom];
    const distance = Math.hypot(...coords.map((value, index) => value - first[index]));
    viewer.removeAllShapes(); refreshLabels();
    const start = {x: first[0], y: first[1], z: first[2]}, end = {x: coords[0], y: coords[1], z: coords[2]};
    viewer.addLine({start, end, color: "#d17b19", dashed: true, linewidth: 3});
    announce(`Separation: ${distance.toFixed(3)} Å between atoms ${firstAtom + 1} and ${atom.index + 1}.`);
    firstAtom = null; viewer.render();
  }
  function renderSample(preserveView = false) {
    if (!viewer) return;
    const previous = preserveView && currentModel ? viewer.getView() : null;
    viewer.removeAllModels(); viewer.removeAllShapes(); viewer.removeAllLabels(); currentModel = null;
    firstAtom = null; details();
    const f = frame(); $("empty").style.display = f ? "none" : "grid";
    $("empty").textContent = conformer()?.error || "No saved coordinates are available for this selection.";
    for (const id of ["xyz", "png", "measure", "spin", "reset", "left", "right", "tilt", "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down"]) $(id).disabled = !f;
    if (!f) { spinning = false; viewer.spin(false); $("spin").setAttribute("aria-pressed", "false"); return; }
    currentModel = viewer.addModel(f.molblock, "sdf", {keepH: true});
    // Molfile stores only four decimals. Restore the original saved precision
    // before rendering, selection, and distance measurement.
    currentModel.selectedAtoms({}).forEach((atom, index) => {
      [atom.x, atom.y, atom.z] = f.coords[index];
    });
    viewer.setClickable({}, true, atomClicked);
    applyStyle();
    if (previous) viewer.setView(previous); else { viewer.zoomTo(); viewer.rotate(15, "x"); viewer.rotate(-15, "y"); viewer.zoom(1.4); }
    viewer.render();
    announce(measuring ? "Click two atoms to measure their separation." : "Click an atom to see its element and saved coordinates.");
  }
  function selectGraph(position) {
    graphPosition = position; conformerPosition = 0;
    const c = candidate();
    document.querySelectorAll(".candidate").forEach((button, index) => {
      button.classList.toggle("active", index === position); button.setAttribute("aria-pressed", String(index === position));
    });
    $("graph").innerHTML = c.svg;
    $("smiles").textContent = c.smiles;
    $("selection-title").textContent = (generated ? "Generated 3D" : "Candidate " + (position + 1)) + " · " + c.formula;
    elementLegend();
    $("conformer").replaceChildren();
    c.conformers.forEach((item, index) => {
      const option = document.createElement("option"); option.value = String(index);
      option.textContent = "Conformer " + (item.conformer_index + 1); $("conformer").append(option);
    });
    $("conformer").disabled = !c.conformers.length;
    $("geometry").value = "final";
    $("geometry").options[1].disabled = !c.conformers.some(item => item.raw);
    renderSample();
  }
  function irPlot() {
    const svg = $("ir-plot"), ns = "http://www.w3.org/2000/svg", input = data.input_ir;
    if (!input?.frequencies?.length) { $("ir-card").hidden = true; return; }
    const maximum = Math.max(...input.intensities, 1), high = Math.max(4000, ...input.frequencies), low = 0;
    const x = frequency => 40 + (high - frequency)/(high - low)*500;
    const y = value => 145 - value/maximum*120;
    function node(tag, attrs, content) { const element = document.createElementNS(ns, tag);
      for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
      if (content !== undefined) element.textContent = content; svg.append(element); }
    for (let tick = 0; tick <= 4; tick++) {
      const frequency = high*tick/4;
      node("line", {x1: x(frequency), x2: x(frequency), y1: 145, y2: 150, stroke: "#92a8b5"});
      node("text", {x: x(frequency), y: 169, "text-anchor": "middle", fill: "#526675", "font-size": 12}, Math.round(frequency));
    }
    node("line", {x1: 40, x2: 540, y1: 145, y2: 145, stroke: "#92a8b5"});
    input.frequencies.forEach((frequency, index) => node("line", {x1: x(frequency), x2: x(frequency),
      y1: 145, y2: y(input.intensities[index]), stroke: "#087f77", "stroke-width": 1.5}));
    node("text", {x: 6, y: 27, fill: "#526675", "font-size": 12}, maximum.toFixed(1));
    node("text", {x: 14, y: 148, fill: "#526675", "font-size": 12}, "0");
  }
  irPlot();
  data.candidates.forEach((c, position) => {
    const button = document.createElement("button"); button.className = "candidate"; button.setAttribute("aria-pressed", "false");
    const rank = document.createElement("span"); rank.className = "rank"; rank.textContent = position + 1;
    const text = document.createElement("div"), name = document.createElement("strong"), detail = document.createElement("span");
    name.textContent = c.formula; detail.textContent = c.conformers.length + (generated ? " generated conformers" : " saved conformers");
    text.append(name, detail); button.append(rank, text); button.onclick = () => selectGraph(position); $("candidates").append(button);
  });
  try {
    viewer = $3Dmol.createViewer($("scene"), {backgroundColor: "#f6f9fb", antialias: true});
    viewer.setProjection("perspective");
  }
  catch (error) { $("empty").style.display = "grid"; $("empty").textContent = "3D rendering could not start. Enable WebGL / graphics acceleration in Chrome or Edge and reopen this file."; }
  if (!viewer || !data.candidates.length) {
    for (const id of ["xyz", "png", "measure", "spin", "reset", "left", "right", "tilt", "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down", "style", "hydrogens", "clear-measure"]) $(id).disabled = true;
  }
  if (viewer && data.candidates.length) selectGraph(0);
  if (!data.candidates.length) { $("empty").style.display = "grid"; $("empty").textContent = "No valid molecular graph was generated. No replacement structure is shown."; details(); }
  $("conformer").onchange = () => { conformerPosition = Number($("conformer").value); renderSample(true); };
  $("geometry").onchange = () => renderSample(true);
  $("style").onchange = applyStyle;
  $("hydrogens").onchange = () => { clearMeasurement(); applyStyle(); };
  $("spin").onclick = () => { if (!frame()) return; spinning = !spinning;
    viewer.spin(spinning ? "y" : false, .65); $("spin").setAttribute("aria-pressed", String(spinning)); };
  $("reset").onclick = () => { spinning = false; viewer?.spin(false); $("spin").setAttribute("aria-pressed", "false"); renderSample(); };
  const rotate = (degrees, axis) => { viewer?.rotate(degrees, axis); viewer?.render(); };
  $("left").onclick = () => rotate(-20, "y"); $("right").onclick = () => rotate(20, "y"); $("tilt").onclick = () => rotate(20, "x");
  $("zoom-in").onclick = () => { viewer?.zoom(1.2); viewer?.render(); };
  $("zoom-out").onclick = () => { viewer?.zoom(1/1.2); viewer?.render(); };
  const pan = (x, y) => { viewer?.translate(x, y); viewer?.render(); };
  $("pan-left").onclick = () => pan(-25, 0); $("pan-right").onclick = () => pan(25, 0);
  $("pan-up").onclick = () => pan(0, 25); $("pan-down").onclick = () => pan(0, -25);
  $("fullscreen").onclick = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $("scene-card").requestFullscreen();
    } catch (_) { announce("Full screen is unavailable here. Open the downloaded HTML directly in Chrome or Edge."); }
  };
  document.addEventListener("fullscreenchange", () => {
    $("fullscreen").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
    viewer?.resize(); viewer?.render();
  });
  $("measure").onclick = () => { measuring = !measuring; $("measure").setAttribute("aria-pressed", String(measuring)); clearMeasurement(); };
  $("clear-measure").onclick = clearMeasurement;
  const filename = suffix => data.source + "_candidate" + (graphPosition + 1) + "_conformer" + (conformer()?.conformer_index + 1) + "_" + $("geometry").value + suffix;
  function download(href, name) { const anchor = document.createElement("a"); anchor.href = href; anchor.download = name; anchor.click(); }
  $("xyz").onclick = () => { const f = frame(); if (!f) return;
    const url = URL.createObjectURL(new Blob([f.xyz], {type: "chemical/x-xyz"})); download(url, filename(".xyz")); setTimeout(() => URL.revokeObjectURL(url), 1000); };
  $("png").onclick = () => { if (!frame()) return; viewer.render(); download(viewer.pngURI(), filename(".png")); };
  new ResizeObserver(() => { viewer?.resize(); viewer?.render(); }).observe($("scene"));
  document.addEventListener("visibilitychange", () => { if (document.hidden && spinning) { spinning = false;
    viewer?.spin(false); $("spin").setAttribute("aria-pressed", "false"); } });
})();
