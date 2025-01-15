#lang rosette

;; 1) 引入已有模块
(require "data-structures.rkt"   ;; 提供 (struct Grid ...)
         "objects.rkt"           ;; 提供 (objects grid univalued? diagonal? without-bg?)
         "json-reader.rkt"       ;; 提供 (read-all-json-files dir)
        ;;;  racket/set
         typed/racket)

;;; (require typed/racket)

;; -------------------------------------------------------------------------
;; 2) 定义符号 DSL (极简示例)：可做 rotate/hmirror/no-op 的组合
;;    在真实项目里，可以放更复杂的 DSL
;; -------------------------------------------------------------------------
(define-type (Expr)
  (U 'NoOp
     (Rot90 Expr)
     (HMirror Expr)))

;; 一个解释器
(define (interp e obj)
  (match e
    ['NoOp
     obj]
    [(Rot90 sub)
     (rotate90 (interp sub obj))]     ;; rotate90 => 你可以在 objects.rkt 或其他文件里实现
    [(HMirror sub)
     (hmirror (interp sub obj))]))    ;; hmirror 同理


;; -------------------------------------------------------------------------
;; 3) 定义一个合成函数，用 Rosette 符号化 e，并让 inputObjects => outputObjects
;; -------------------------------------------------------------------------

(define (synthesize-transformation input-obj output-obj)
  ;; 先定义符号变量 e
  (define-symbolic e (Expr))

  ;; 加入约束：interp(e, input-obj) 与 output-obj 相等
  (assert (equal? (interp e input-obj) output-obj))

  ;; 调用 solve
  (define result (solve))
  (if (sat? result)
      (begin
        (displayln "Found a transformation!")
        (define m (solution result))
        (define final-e (lookup m e))  ;; e的具体AST
        (displayln (format " => ~a" final-e))
        final-e)
      (begin
        (displayln "No solution found.")
        #f)))

;; -------------------------------------------------------------------------
;; 4) 主流程：读取 JSON => 提取 input/output grid => 转成 objects => 合成
;; -------------------------------------------------------------------------

(define (main dir)
  ;; 4.1 读取此目录下所有 JSON
  (define all-json (read-all-json-files dir))  ;; => list of JSON data
  (for ([json-data (in-list all-json)])
    (displayln "---------------------")
    (displayln (format "Now process: ~a" json-data))

    ;; 取 "train" / "test"
    (define train-data (hash-ref json-data 'train))
    (define test-data  (hash-ref json-data 'test))

    ;; 简化：只处理 train-data 里的第一个 pair
    (define first-pair (if (null? train-data) #f (car train-data)))
    (when first-pair
      ;; 把 input/output => Grid => objects
      (define input-grid  (Grid (hash-ref first-pair 'input)))
      (define output-grid (Grid (hash-ref first-pair 'output)))

      (define input-objects  (objects input-grid  #f #f #f))  ;; univalued? #f, diagonal? #f, without-bg? #f
      (define output-objects (objects output-grid #f #f #f))

      ;; 假设只取第1个对象 => 真实情况里看你要怎么处理
      (define in-obj  (if (set-empty? input-objects)
                          (set)
                          (car (set->list input-objects))))
      (define out-obj (if (set-empty? output-objects)
                          (set)
                          (car (set->list output-objects))))

      (displayln (format "Input obj=~a" in-obj))
      (displayln (format "Output obj=~a" out-obj))

      ;; 尝试合成一个 DSL 表达式 e
      (synthesize-transformation in-obj out-obj))))

;; -------------------------------------------------------------------------
;; 5) 程序入口：可在命令行 racket program-synthesis.rkt <dir>
;; -------------------------------------------------------------------------
(provide main)

;;; 四、后续要加的新函数或判断？
;;; 如果你要添加更多变换（如 rotate180, cmirror, dmirror），只需在 DSL 里增加对应的构造 (Rotate180 e), (CMirror e)，并在 interp 里实现 (match (Rotate180 sub) ...) => (rotate180 (interp sub obj))。
;;; 如果要添加判断（例如「若对象颜色 = 7 则 hmirror 否则 rotate90」），就需要在 DSL 里定义 (IfHasColor c e1 e2) 之类，然后在 interp 里做 (if (has-color? obj c) (interp e1 obj) (interp e2 obj))。
;;; 每次新加语法构造，就能让 Rosette 搜索到更丰富的程序空间。相应地，你也要 (assert ...) 期望在多个例子上都成立，就能合成更复杂的“程序化变换”。

