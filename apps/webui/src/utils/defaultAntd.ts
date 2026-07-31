export const DEFAULT_ANTD_CLASS_NAME = "default-antd";
export const DEFAULT_ANTD_QUERY_PARAM = "defaultAntd";

export function isDefaultAntdMode() {
  return new URLSearchParams(window.location.search).get(DEFAULT_ANTD_QUERY_PARAM) === "1";
}

export function applyDefaultAntdModeClass() {
  document.documentElement.classList.toggle(DEFAULT_ANTD_CLASS_NAME, isDefaultAntdMode());
}
