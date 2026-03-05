# 操作说明

请使用如下命令将 Markdown 文件导出为 Word（.docx）文件：

```
pandoc tennis_fomo_thesis.md -o tennis_fomo_thesis.docx --resource-path=.
```

- 该命令会自动将图片等资源一并导出到 Word 文档中。
- 请确保已安装 pandoc 工具（https://pandoc.org/）。
- 如需自定义样式，可添加 `--reference-doc=模板.docx` 参数。

如需自动化脚本或遇到问题可随时告知。
