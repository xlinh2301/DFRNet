const pageSize = 24;
let page = 1;
const selected = new Set();

const grid = document.getElementById("grid");
const pageInfo = document.getElementById("pageInfo");
const selectedCount = document.getElementById("selectedCount");

function segSvg(segmentation, width, height) {
  const pts = segmentation[0];
  const points = [];
  for (let i = 0; i < pts.length; i += 2) points.push(`${pts[i]},${pts[i+1]}`);
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <polygon points="${points.join(' ')}" />
  </svg>`;
}

function buildQuery() {
  const params = new URLSearchParams({ page, page_size: pageSize, sort: "yolo_conf" });
  params.set("order", document.getElementById("order").value);
  const minConf = document.getElementById("minConf").value;
  const maxConf = document.getElementById("maxConf").value;
  if (minConf !== "") params.set("min_conf", minConf);
  if (maxConf !== "") params.set("max_conf", maxConf);
  return params.toString();
}

async function loadPage() {
  const res = await fetch(`/api/supplement/candidates?${buildQuery()}`);
  const data = await res.json();
  const totalPages = Math.max(1, Math.ceil(data.total_candidates / pageSize));
  pageInfo.textContent = `Page ${data.page} / ${totalPages} (${data.total_candidates} candidates)`;
  render(data.items);
}

function render(items) {
  grid.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("div");
    card.className = "card";

    const imgwrap = document.createElement("div");
    imgwrap.className = "imgwrap";
    imgwrap.innerHTML = `<img src="${item.url}" alt="${item.file_name}" loading="lazy" />${segSvg(item.segmentation, item.width, item.height)}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "candidate-select";
    checkbox.checked = selected.has(item.annotation_id);
    checkbox.onchange = () => {
      if (checkbox.checked) selected.add(item.annotation_id);
      else selected.delete(item.annotation_id);
      selectedCount.textContent = `${selected.size} selected`;
    };
    imgwrap.appendChild(checkbox);
    card.appendChild(imgwrap);

    const fname = document.createElement("div");
    fname.className = "fname";
    fname.textContent = item.file_name;
    card.appendChild(fname);

    const text = document.createElement("div");
    text.className = "gt-text";
    text.innerHTML = `<b>text</b> "${item.text}"`;
    card.appendChild(text);

    const conf = document.createElement("div");
    conf.className = "conf";
    conf.textContent = `conf ${item.yolo_conf?.toFixed?.(3) ?? "n/a"}  ·  angle ${item.angle ?? "n/a"}`;
    card.appendChild(conf);

    grid.appendChild(card);
  }
}

document.getElementById("applyFilters").onclick = () => {
  page = 1;
  loadPage();
};

document.getElementById("prevBtn").onclick = () => {
  if (page > 1) {
    page -= 1;
    loadPage();
  }
};
document.getElementById("nextBtn").onclick = () => {
  page += 1;
  loadPage();
};

document.getElementById("importBtn").onclick = async () => {
  if (selected.size === 0) return;
  const res = await fetch("/api/supplement/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ annotation_ids: Array.from(selected) }),
  });
  const data = await res.json();
  alert(`Imported: ${data.imported.length}, Skipped: ${data.skipped.length}`);
  selected.clear();
  selectedCount.textContent = "0 selected";
  loadPage();
};

loadPage();
