/// <reference types="mdast" />
import { h } from "hastscript";

/**
 * Creates a PDF viewer component with embedded styling.
 *
 * @param {Object} _properties - The properties of the component (unused).
 * @param {import('mdast').RootContent[]} children - The children elements containing PDF name.
 * @returns {import('mdast').Parent} The created PDF viewer component.
 */
export function PDFViewerComponent(_properties, children) {
	if (!Array.isArray(children) || children.length === 0) {
		return h(
			"div",
			{ class: "hidden" },
			"Invalid PDF directive. Usage: :::pdf\nfilename.pdf\n:::",
		);
	}

	// 获取 PDF 文件名
	let pdfName = "";
	if (children[0]?.children?.[0]) {
		pdfName = children[0].children[0].value.trim();
	}

	if (!pdfName) {
		return h("div", { class: "pdf-error-message" }, "PDF 文件名不能为空");
	}

	// 构建 PDF 路径
	const pdfPath = `/blog/assets/papers/${pdfName}`;

	// 创建 PDF 预览组件
	return h("div", { class: "pdf-viewer-container" }, [
		h("div", { class: "pdf-viewer-card" }, [
			h("iframe", {
				src: pdfPath,
				width: "100%",
				style: "display: block; border: none;",
				loading: "lazy",
				title: `${pdfName} 预览`,
			}),
		]),
		h("div", { class: "pdf-actions" }, [
			h(
				"a",
				{
					href: pdfPath,
					target: "_blank",
					class: "action-button",
				},
				[h("span", {}, "🔗"), " 新窗口打开"],
			),
			h(
				"a",
				{
					href: pdfPath,
					download: true,
					class: "action-button",
				},
				[h("span", {}, "⬇️"), " 下载论文"],
			),
		]),
	]);
}
