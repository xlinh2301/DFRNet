const pageSize = 24;
let page = 1;

const grid = document.getElementById("grid");
const pageInfo = document.getElementById("pageInfo");
const emptyNote = document.getElementById("emptyNote");

function bboxSvg(bbox, width, height) {
  const [x, y, w, h] = bbox;
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <rect x="${x}" y="${y}" width="${w}" height="${h}" />
  </svg>`;
}

async function loadPage() {
  const res = await fetch(`/api/eval/mismatches?page=${page}&page_size=${pageSize}`);
  const data = await res.json();
  const totalPages = Math.max(1, Math.ceil(data.total_mismatches / pageSize));
  pageInfo.textContent = `Page ${data.page} / ${totalPages} (${data.total_mismatches} mismatches)`;
  emptyNote.style.display = data.total_mismatches === 0 ? "block" : "none";
  render(data.items);
}

function render(items) {
  grid.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("div");
    card.className = "card";

    const imgwrap = document.createElement("div");
    imgwrap.className = "imgwrap";
    imgwrap.innerHTML = `<img src="${item.url}" alt="${item.file_name}" />${bboxSvg(item.bbox, item.width, item.height)}`;
    card.appendChild(imgwrap);

    const fname = document.createElement("div");
    fname.textContent = item.file_name;
    fname.style.fontSize = "0.75rem";
    fname.style.color = "#888";
    card.appendChild(fname);

    const gt = document.createElement("div");
    gt.textContent = `GT: "${item.gt_text}"`;
    card.appendChild(gt);

    const pred = document.createElement("div");
    pred.textContent = `PPOCRv5: "${item.predicted_text}"`;
    pred.style.color = "#f66";
    card.appendChild(pred);

    const row = document.createElement("div");
    row.className = "ann-row";

    const input = document.createElement("input");
    input.type = "text";
    input.value = item.gt_text;

    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    saveBtn.onclick = async () => {
      await fetch(`/api/review/annotations/${item.annotation_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input.value }),
      });
      loadPage();
    };

    const delBtn = document.createElement("button");
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!confirm(`Delete annotation for "${item.file_name}"?`)) return;
      await fetch(`/api/review/annotations/${item.annotation_id}`, { method: "DELETE" });
      loadPage();
    };

    row.appendChild(input);
    row.appendChild(saveBtn);
    row.appendChild(delBtn);
    card.appendChild(row);

    grid.appendChild(card);
  }
}

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

loadPage();
