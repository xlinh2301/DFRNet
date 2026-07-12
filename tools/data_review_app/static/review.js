const pageSize = 24;
let page = 1;

const grid = document.getElementById("grid");
const pageInfo = document.getElementById("pageInfo");

function segSvg(segmentation, width, height) {
  const pts = segmentation[0];
  const points = [];
  for (let i = 0; i < pts.length; i += 2) points.push(`${pts[i]},${pts[i+1]}`);
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <polygon points="${points.join(' ')}" />
  </svg>`;
}

async function loadPage() {
  const res = await fetch(`/api/review/images?page=${page}&page_size=${pageSize}`);
  const data = await res.json();
  const totalPages = Math.max(1, Math.ceil(data.total_images / pageSize));
  pageInfo.textContent = `Page ${data.page} / ${totalPages} (${data.total_images} images)`;
  render(data.items);
}

function render(items) {
  grid.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("div");
    card.className = "card";

    const overlays = item.annotations.map(a => segSvg(a.segmentation, item.width, item.height)).join("");
    const imgwrap = document.createElement("div");
    imgwrap.className = "imgwrap";
    imgwrap.innerHTML = `<img src="${item.url}" alt="${item.file_name}" loading="lazy" />${overlays}`;
    card.appendChild(imgwrap);

    const fname = document.createElement("div");
    fname.className = "fname";
    fname.textContent = item.file_name;
    card.appendChild(fname);

    for (const ann of item.annotations) {
      const row = document.createElement("div");
      row.className = "ann-row";

      const input = document.createElement("input");
      input.type = "text";
      input.value = ann.text;

      const saveBtn = document.createElement("button");
      saveBtn.className = "save-btn";
      saveBtn.textContent = "Save";
      saveBtn.onclick = async () => {
        try {
          const res = await fetch(`/api/review/annotations/${ann.annotation_id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: input.value }),
          });
          if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
          saveBtn.textContent = "Saved";
          setTimeout(() => (saveBtn.textContent = "Save"), 800);
        } catch (err) {
          alert(`Save failed: ${err.message}`);
        }
      };

      const delBtn = document.createElement("button");
      delBtn.className = "del-btn";
      delBtn.textContent = "Delete";
      delBtn.onclick = async () => {
        if (!confirm(`Delete annotation for "${item.file_name}"?`)) return;
        try {
          const res = await fetch(`/api/review/annotations/${ann.annotation_id}`, { method: "DELETE" });
          if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
          loadPage();
        } catch (err) {
          alert(`Delete failed: ${err.message}`);
        }
      };

      row.appendChild(input);
      row.appendChild(saveBtn);
      row.appendChild(delBtn);
      card.appendChild(row);
    }

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
