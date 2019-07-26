"""
tables = ["Bound1.tex", "Base1.tex", "Scale1.tex", "Bound3.tex", "Base3.tex",
            "Scale3.tex", "Bound5.tex", "Base5.tex", "Scale5.tex",
            "Bound10.tex", "Base10.tex", "Scale10.tex",
            "Bound100.tex", "Base100.tex", "Scale100.tex", "Base1000.tex", "CG.tex"]
key = [:Bound1, :Base1, :Scale1, :Bound3, :Base3,
             :Scale3, :Bound5, :Base5, :Scale5,
           :Bound10, :Base10, :Scale10,
            :Bound100, :Base100, :Scale100, :Base1000, :CG]
"""

tables = ["Base1.tex", "CG.tex"]
key = [:Base1, :CG]

for i in 1:length(key)
    df = get(stats, key[i], false)
    df1 = df[:, [2, 6, 7, 8, 9, 12, 13, 21]]
    open(tables[i], "w") do io
        latex_table(io, df1)
    end
end

score(h) = h.elapsed_time
performance_profile(stats, score)
