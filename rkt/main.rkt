#lang racket

(require "json-reader.rkt"
         "process-data.rkt")

;; ----------------------------------------------------------------------
;; 主程序入口
;; ----------------------------------------------------------------------

(define (main)
  "主程序：读取目录下所有 JSON 文件并处理。"
  (define dir-path "/Users/zhangdexiang/github/VSAHDC/arc-dsl/rkt/training-data") ; 替换为你的 JSON 文件目录路径
  (define all-json-data (read-all-json-files dir-path))

  ;; 循环处理每个 JSON 数据
  (for ([json-data all-json-data])
    (process-json-data json-data)))

;; 执行主程序
(main)
