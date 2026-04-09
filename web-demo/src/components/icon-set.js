const ICONS = {
  guide: `
    <path d="M4 5h16v14H4z" />
    <path d="M8 9h8M8 13h6" />
  `,
  dataset: `
    <rect x="3" y="4" width="18" height="5" rx="1.5" />
    <rect x="3" y="10" width="18" height="5" rx="1.5" />
    <rect x="3" y="16" width="18" height="5" rx="1.5" />
    <path d="M7 6.5h0M7 12.5h0M7 18.5h0" />
  `,
  train: `
    <circle cx="7" cy="7" r="3" />
    <circle cx="17" cy="7" r="3" />
    <circle cx="12" cy="17" r="3" />
    <path d="M9.5 8.5l3 6M14.5 8.5l-3 6M9.8 7h4.4" />
  `,
  evaluate: `
    <path d="M4 4h16v16H4z" />
    <path d="M8 16l3-3 2 2 3-4" />
    <path d="M8 8h8" />
  `,
  apply: `
    <path d="M12 3v13" />
    <path d="M8 12l4 4 4-4" />
    <rect x="4" y="18" width="16" height="3" rx="1.5" />
  `,
  info: `
    <circle cx="12" cy="12" r="9" />
    <path d="M12 10v6M12 7h0" />
  `,
  warning: `
    <path d="M12 3l9 17H3z" />
    <path d="M12 9v5M12 17h0" />
  `,
  metric: `
    <path d="M4 19h16" />
    <rect x="6" y="11" width="3" height="6" />
    <rect x="11" y="8" width="3" height="9" />
    <rect x="16" y="6" width="3" height="11" />
  `,
};

export function iconSvg(name, className = "inline-icon", title = "") {
  const path = ICONS[name] || ICONS.info;
  const titleMarkup = title ? `<title>${title}</title>` : "";
  return `
    <svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      ${titleMarkup}
      ${path}
    </svg>
  `;
}
