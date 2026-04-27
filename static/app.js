const navToggle = document.querySelector("[data-nav-toggle]");
const navMenu = document.querySelector("[data-nav-menu]");

if (navToggle && navMenu) {
    navToggle.addEventListener("click", () => {
        navMenu.classList.toggle("is-open");
    });
}

const heightInput = document.querySelector("[data-height]");
const weightInput = document.querySelector("[data-weight]");
const bmiInput = document.querySelector("[data-bmi]");

function updateBmi() {
    if (!heightInput || !weightInput || !bmiInput) {
        return;
    }

    const heightCm = Number.parseFloat(heightInput.value);
    const weightKg = Number.parseFloat(weightInput.value);

    if (!heightCm || !weightKg) {
        return;
    }

    const heightM = heightCm / 100;
    const bmi = weightKg / (heightM * heightM);
    bmiInput.value = bmi.toFixed(1);
}

if (heightInput && weightInput && bmiInput) {
    heightInput.addEventListener("input", updateBmi);
    weightInput.addEventListener("input", updateBmi);
}
