

;;; deepseek r1



#lang racket

(require rosette)
(require (only-in rosette/lib/synthax define-synthax))

;; ======================
;; 基础类型与网格操作
;; ======================

(define (parse-grid str)
  (eval (read (open-input-string (string-replace str " " "")))))

(define (grid-equal? g1 g2)
  (equal? g1 g2))

(struct object (color positions) #:transparent)

;; ======================
;; 修正后的符号化DSL原语定义
;; ======================

(define-synthax (primitive-op input-grid) ; 移除带默认值的参数
  #:base (choose
          input-grid
          (fill input-grid (?? integer?) (?? (listof (listof integer?)))))
  #:else (let ([prev (primitive-op input-grid)]) ; 移除递归深度参数
           (choose
            prev
            (fill prev (?? integer?) (?? (listof (listof integer?))))
            (filter-color prev (?? integer?))
            (extract-objects prev #:without-bg? (?? boolean?)))))

;; ======================
;; 核心原语实现
;; ======================

(define (extract-objects grid #:without-bg? [without-bg? #t])
  (if without-bg?
      (filter (λ (obj) (not (= (object-color obj) 0))) (symbolic-extract grid))
      (symbolic-extract grid)))

(define (symbolic-extract grid) ; 添加最小实现
  (list (object 4 '((1 1)(1 2)(2 1)))))

(define (filter-color objs color)
  (filter (λ (obj) (= (object-color obj) color)) objs))

(define (fill grid value patch)
  (for/list ([row grid] [i (in-naturals)])
    (for/list ([cell row] [j (in-naturals)])
      (if (member (list i j) (flatten patch)) value cell))))

;; ======================
;; Rosette符号化求解引擎
;; ======================

(define-symbolic grid-sym integer?)
(define-symbolic op-sym integer?)

(define (solve-task input-output-pairs)
  (define sol
    (synthesize
     #:forall (list grid-sym op-sym)
     #:guarantee
     (for/all ([io input-output-pairs])
       (match-let ([(cons input output) io])
         (assert (grid-equal? (primitive-op input) output))))))
  (evaluate (primitive-op (car (car input-output-pairs))) sol))

;; ======================
;; 示例任务与测试
;; ======================

(define input1 (parse-grid "[[0 0 0 0 0]
                            [0 4 4 4 0]
                            [0 4 0 4 0]
                            [0 4 4 4 0]
                            [0 0 0 0 0]]"))

(define output1 (parse-grid "[[0 0 0 0 0]
                             [0 4 4 4 0]
                             [0 4 4 4 0]
                             [0 4 4 4 0]
                             [0 0 0 0 0]]"))

(define solution (solve-task (list (cons input1 output1))))

(printf "Found solution:\n~a\n" solution)
(printf "Verification: ~a\n" (grid-equal? (solution input1) output1))