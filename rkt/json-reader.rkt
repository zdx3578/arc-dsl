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
(define (read-json-file filepath filename)
  "读取指定路径的 JSON 文件并解析为 Racket 数据结构，同时将文件名添加到数据中。"
  (with-handlers ([exn:fail? (lambda (e)
                               (displayln (format "Failed to read JSON file: ~a" filepath))
                               (displayln (exn-message e))
                               #f)])
    (define json-str (file->string filepath))
    (define ip (open-input-string json-str))
    (define json-data (read-json ip))
    ;; Add the filename as a key to the JSON data
    (hash-set json-data 'filepath filepath)
    (hash-set json-data 'filename filename)
    )) ; No need for path->string here

;; 遍历目录并读取所有 JSON 文件
;; 参数：
;; - dir-path: string（目录路径）
;; 返回：
;; - 解析后的 JSON 数据列表
(define (read-all-json-files dir-path)
  "遍历指定目录，读取所有 JSON 文件并返回解析后的数据列表，每个数据包含文件名。"
  (define files (directory-list dir-path))
  (define json-files
    (filter (λ (f)
              (let ([f-str (path->string f)]) ; Convert path to string for comparison
                (string-ci-suffix? ".json" (string-downcase f-str))))
            files))
  ;; 读取 JSON 文件并返回包含文件名的数据
  (map (λ (f)
         (read-json-file (path->string (build-path dir-path f)) f  )) ; Convert path to string
       json-files))