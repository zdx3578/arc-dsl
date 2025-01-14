#lang rosette

(require rosette/lib/match)

;; 1) DSL 定义
(struct NoOp ())
(struct Rot90 (sub))
(struct HMirror (sub))
(struct Compose (e1 e2))

;; 2) 假设已有 rotate90/hmirror 函数
(define (rotate90 obj)
  obj) ;; TODO: 你的实际实现
(define (hmirror obj)
  obj) ;; TODO: 你的实际实现

;; 3) 解释器
(define (interp expr obj)
  (match expr
    [(NoOp) 
     obj]
    [(Rot90 sub)
     (rotate90 (interp sub obj))]
    [(HMirror sub)
     (hmirror (interp sub obj))]
    [(Compose e1 e2)
     (interp e2 (interp e1 obj))]))

;; 4) 符号化 + 约束
;;   为示例起见，只允许 e ∈ { 'NoOp, 'Rot90, 'HMirror} (不含子表达式).
(define-symbolic choice SymbolicChoice)
(assert (or (eq? choice 'no-op)
            (eq? choice 'rot90)
            (eq? choice 'hmirror)))

(define (make-dsl c)
  (cond
    [(eq? c 'no-op)    (NoOp)]
    [(eq? c 'rot90)    (Rot90 (NoOp))]
    [(eq? c 'hmirror)  (HMirror (NoOp))]
    [else (error "???")]))

;; 假设我们有 input-obj, output-obj
(define input-obj
  (set '( (3 (0 0))
          (3 (0 1)) )))
(define output-obj
  (set '( (3 (0 0))
          (3 (0 1)) ))  ;; 这里示例, same as input

;; 约束: interp DSL on input-obj => output-obj
(assert (equal? (interp (make-dsl choice) input-obj) output-obj))

;; 求解
(define result (solve))
(if (sat? result)
    (begin
      (displayln "SAT Found!")
      (define sol (solution result))
      (displayln (format "choice=~a" (hash-ref sol 'choice))))
    (displayln "UNSAT or Unknown"))

