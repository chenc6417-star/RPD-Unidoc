
const exampleImages = {
  'gt': 'assets/example_gt.png',
  'qwen-base': 'assets/example_qwen.png',
  'full-ft': 'assets/example_full_ft.png',
  'sara': 'assets/example_sara.png'
};

function switchExample(type) {
  const imgElement = document.getElementById('example-img');
  if (imgElement && exampleImages[type]) {
    imgElement.src = exampleImages[type];
  }

  
  const buttons = document.querySelectorAll('.tab-btn');
  buttons.forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
}


document.addEventListener("DOMContentLoaded", () => {
  const copyBtn = document.getElementById("copy-btn");
  const bibtexCode = document.getElementById("bibtex-code");

  if (copyBtn && bibtexCode) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(bibtexCode.innerText).then(() => {
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        setTimeout(() => {
          copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy BibTeX';
        }, 2000);
      });
    });
  }
});
