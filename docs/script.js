document.addEventListener("DOMContentLoaded", () => {
  // 1. Fetch & populate benchmark results
  const tableBody = document.querySelector("#results-table tbody");

  fetch("results.json")
    .then((response) => response.json())
    .then((data) => {
      tableBody.innerHTML = "";
      data.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${row.setting}</td>
          <td><strong>${row.model}</strong></td>
          <td>${row.trainable_params || "—"}</td>
          <td>${row.dla_f1 !== null ? row.dla_f1.toFixed(4) : "—"}</td>
          <td>${row.dla_miou !== null ? row.dla_miou.toFixed(4) : "—"}</td>
          <td>${row.ocr_acc !== null ? row.ocr_acc.toFixed(4) : "—"}</td>
          <td>${row.ro_score !== null ? row.ro_score.toFixed(4) : "—"}</td>
          <td>${row.meteor !== null ? row.meteor.toFixed(4) : "—"}</td>
          <td>${row.bert_score !== null ? row.bert_score.toFixed(4) : "—"}</td>
        `;
        tableBody.appendChild(tr);
      });
    })
    .catch((error) => {
      console.error("Error loading results.json:", error);
    });

  // 2. BibTeX copy button
  const copyBtn = document.getElementById("copy-btn");
  const bibtexCode = document.getElementById("bibtex-code");

  if (copyBtn && bibtexCode) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(bibtexCode.innerText).then(() => {
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        setTimeout(() => {
          copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
        }, 2000);
      });
    });
  }
});
