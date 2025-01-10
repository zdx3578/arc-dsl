;; json-reader.rkt
#lang rosette

(require "data-structures.rkt"
         json
         racket/file
         racket/path
         racket/string) ; 引入 string 模块

(provide read-json-file read-all-json-files)

;; 定义不区分大小写的后缀检查函数
(define (string-ci-suffix? suffix str)
  "检查字符串 str 是否以 suffix 结尾，不区分大小写。"
  (let ([len-suffix (string-length suffix)]
        [len-str (string-length str)])
    (and (>= len-str len-suffix)
         (string-ci=? (substring str (- len-str len-suffix)) suffix))))

;; 读取并解析单个 JSON 文件
;; 参数：
;; - filepath: string
;; 返回：
;; - 解析后的 JSON 数据（哈希表等）
(define (read-json-file filepath)
  "读取指定路径的 JSON 文件并解析为 Racket 数据结构。"
  (displayln (format "Reading JSON file: ~a" filepath)) ; 使用 displayln 进行打印
  (define json-str (file->string filepath))
  (define ip (open-input-string json-str))
  (read-json ip))

;; 遍历目录并读取所有 JSON 文件
;; 参数：
;; - dir-path: string（目录路径）
;; 返回：
;; - 解析后的 JSON 数据列表
(define (read-all-json-files dir-path)
  "遍历指定目录，读取所有 JSON 文件并返回解析后的数据列表。"
  ; (displayln (format "Listing files in directory: ~a" dir-path))
  (define files (directory-list dir-path))
  ; (displayln (format "Files found: ~a" files))
  (define json-files
    (filter (λ (f)
              (let ([f-str (path->string f)])
                ; (displayln (format "Checking file: ~a" f-str))
                (string-ci-suffix? ".json" f-str))) ; 使用自定义的不区分大小写的后缀检查
            files))
  ; (displayln (format "JSON files filtered: ~a" json-files))
  (map read-json-file json-files))
