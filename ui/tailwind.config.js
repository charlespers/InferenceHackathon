/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    // Conifer studio is a square, TUI-inspired surface: no rounded corners.
    borderRadius: {
      none: "0px",
      DEFAULT: "0px",
      sm: "0px",
      md: "0px",
      lg: "0px",
      xl: "0px",
      "2xl": "0px",
      full: "9999px", // status dots only
    },
    extend: {
      fontFamily: {
        display: ['"Inter Tight"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ['"Inter Tight"', "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ['"Fraunces"', "ui-serif", "Georgia", "serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        canvas: "var(--studio-canvas)",
        paper: "var(--studio-paper)",
        sunk: "var(--studio-sunk)",
        card: "var(--studio-card)",
        "card-hover": "var(--studio-card-hover)",
        ink: "var(--studio-ink)",
        "ink-soft": "var(--studio-ink-soft)",
        "ink-mute": "var(--studio-ink-mute)",
        "ink-faint": "var(--studio-ink-faint)",
        rule: "var(--studio-rule)",
        "rule-strong": "var(--studio-rule-strong)",
        conifer: "var(--conifer)",
        "conifer-soft": "var(--conifer-soft)",
        baseline: "var(--baseline)",
        steel: "var(--studio-cyan)",
        warn: "var(--studio-warn)",
        fail: "var(--studio-fail)",
      },
      borderColor: { DEFAULT: "var(--studio-rule)" },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.45s cubic-bezier(0.22,1,0.36,1) both",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
