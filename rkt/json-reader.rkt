;; json-reader.rkt
#lang rosette

(require "data-structures.rkt"
         json
         racket/file)

(provide read-json-file read-all-json-files)

;; 读取并解析单个 JSON 文件
;; 参数：
;; - filepath: string
;; 返回：
;; - 解析后的 JSON 数据（哈希表等）
(define (read-json-file filepath)
  "读取指定路径的 JSON 文件并解析为 Racket 数据结构。"
  (define json-str (file->string filepath))
  (read-json json-str))

;; 遍历目录并读取所有 JSON 文件
;; 参数：
;; - dir-path: string（目录路径）
;; 返回：
;; - 解析后的 JSON 数据列表
(define (read-all-json-files dir-path)
  "遍历指定目录，读取所有 JSON 文件并返回解析后的数据列表。"
  (define files (directory-list dir-path))
  (define json-files
    (filter (λ (f) (string-suffix? ".json" f)) files))
  (map read-json-file json-files))
