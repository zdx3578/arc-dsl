;; -------------------------------------------------------
;; 6) 主函数 main
;; -------------------------------------------------------
(define (main dir)
  (define all-json (read-all-json-files dir))  ;; => list of JSON data

  (let ([file-iter 0])
    (for ([json-data (in-list all-json)])
      (set! file-iter (add1 file-iter))

      ;; 显示文件序号和文件名
      (displayln "=================================================================")
      (displayln (format "=================================================================File-Iteration #~a | filename: ~a"
                         file-iter
                         (hash-ref json-data 'filename)))

      ;; 读取 train/test
      (define train-data (hash-ref json-data 'train))
      (define test-data  (hash-ref json-data 'test))

      (let ([pair-iter 0])
        (for ([pair (in-list train-data)])
          (set! pair-iter (add1 pair-iter))

          (displayln "-----------------------------------------------------------------")
          (displayln
           (format " =================================================================File-Iteration #~a  ~a ------- Now processing train pair #: ~a"
                   file-iter
                   (hash-ref json-data 'filename)
                   pair-iter))

          ;; 取出 input-grid, output-grid
          (define input-grid  (Grid (hash-ref pair 'input)))
          (define output-grid (Grid (hash-ref pair 'output)))

          ;; -- 1) 对 input-grid 进行 8 种参数组合 -> 并集
          (define input-obj-set (all-objects-from-grid input-grid))
          (define input-00shapes-set (all-objects-00shape-from-objs input-obj-set))
          (displayln (format "  input-obj-set count = ~a" (set-count input-obj-set)))
          (displayln (format "  input-00shapes-set count = ~a" (set-count input-00shapes-set)))

          ;; -- 2) 对 output-grid 进行 8 种参数组合 -> 并集
          (define output-obj-set (all-objects-from-grid output-grid))
          (define output-00shapes-set (all-objects-00shape-from-objs output-obj-set))
          (displayln (format "  output-obj-set count = ~a" (set-count output-obj-set)))
          (displayln (format "  output-00shapes-set count = ~a" (set-count output-00shapes-set)))


          ;; 这几行只是演示diff之类，如不需要可略过
          (define diff1 (set-subtract set1 set2))
          (define diff2 (set-subtract set1 set2))
          (define diff  (set-subtract diff1 diff2 ))

          ;; -------------------------------------------------------
          ;; 现在对 output-grid 的 8 种组合分别处理（外层循环）
          ;; 一旦找到一个 out-param 能匹配所有 out-obj, 即可停止.
          ;; -------------------------------------------------------
          (let ([outer-iter 0]
                [param-found? #f])   ;; 标记是否已有成功的param

            (for ([out-param (in-list param-combinations)])
              (unless param-found?   ;; 如果已经找到过成功param，就不要再进来了
                (set! outer-iter (add1 outer-iter))

                ;; 取得对当前 out-param 的所有 out-obj
                (define out-obj-set (objects-with-params output-grid out-param))

                (displayln "  ")
                (displayln "  ")
                (displayln
                 (format "-----------------------------------------File #~a ~a train pair #:~a-------------outobj param-iteration ~a---- Output param = ~a, count = ~a"
                         file-iter
                         (hash-ref json-data 'filename)
                         pair-iter
                         outer-iter
                         out-param
                         (set-count out-obj-set)))

                ;; 中层循环：对当前 out-param 下的 out-obj 全部尝试
                (let ([mid-iter 0]
                      [all-out-obj-solved? #t])  ;; 标记“此 out-param 是否能匹配所有 out-obj”

                  (for ([out-obj (in-set out-obj-set)])
                    (when all-out-obj-solved?
                      (set! mid-iter (add1 mid-iter))
                      (displayln "  ")
                      (displayln "  ")
                      (displayln
                       (format "------------------File #~a -- train pair #: ~a--outobj param- ~a--------------out-obj iteration ~a------ Checking out-obj = ~a"
                               file-iter
                               pair-iter
                               outer-iter
                               mid-iter
                               out-obj))

                      ;; 最内层：遍历 input-obj-set, 找到一个成功则跳出
                      (let ([inner-iter 0]
                            [found-one? #f]) ;; 记录是否找到至少一个 in-obj 成功
                        (for ([in-obj (in-set input-obj-set)])
                          (unless found-one?
                            (set! inner-iter (add1 inner-iter))
                            ;; 这里可以加详细日志
                            ;; (displayln (format "      inner-iter ~a => in-obj = ~a" inner-iter in-obj))

                            (when (synthesize-transformation in-obj out-obj)
                              ;; 有一个成功即可跳出最内层
                              (set! found-one? #t)))))

                      ;; 如果该 out-obj 没有任何一个 in-obj 成功匹配，则此 out-param 失败
                      (unless found-one?
                        (set! all-out-obj-solved? #f)))))

                  ;; 如果该 out-param 成功匹配所有 out-obj，就不用再看后续 param
                  (when all-out-obj-solved?
                    (displayln (format "====> Great! Param ~a has solved **all** out-obj for pair #:~a. Stop searching further param." out-param pair-iter))
                    (set! param-found? #t)))))

            (when param-found?
              (displayln (format "====> We found a param (#~a) that solves everything. End param loop early for pair #:~a" outer-iter pair-iter)))))

      (displayln "Done!")))

(provide main)


