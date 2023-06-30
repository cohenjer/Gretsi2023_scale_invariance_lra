import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from shootout.methods.post_processors import median_or_mean_and_errors
pio.templates.default= "plotly_white"
pio.kaleido.scope.mathjax = None

# templating
my_template = go.layout.Template()
my_template.layout=dict(
    font_size = 14,
    width=600*1.62/2, # in px
    height=400*2,
    legend=dict(orientation="h", yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1,
        ),
)

pio.templates["my_template"]=my_template
pio.templates.default = "plotly_white+my_template"

#name = "./results/xp_12_27-03-2023"
name = "./results/xp_29-06-2023"


df = pd.read_pickle(name)
nb_seeds = df["seed"].max()+1 # get nbseed from data

# Averaging and Error bars
df_mean_sparsity = median_or_mean_and_errors(df, "sparsity", ["algorithm","sp","xp"])
df_mean_fms = median_or_mean_and_errors(df, "fms", ["algorithm","sp","xp"])
df_mean_finalerr = median_or_mean_and_errors(df, "final_errors", ["algorithm","sp","xp"])



# Plotting sparsity vs reg value
fig1 = px.line(df_mean_sparsity,
    x="sp",
    y="sparsity",
    log_x=True,
    color="algorithm",
    line_dash="algorithm",
    facet_row="xp",
    error_y="sparsity_08",
    error_y_minus="sparsity_02",
    #template=my_template,
    )
fig1.add_hline(y=0.3*df["rank"][0]*df["dim"][0], row=1, annotation_text="Parcimonie cible", line=dict(dash='dot'))
fig1.add_hline(y=df["rank"][0]*df["dim"][0], row=0, annotation_text="Parcimonie cible", line=dict(dash='dot'))
fig1.update_yaxes(title_text="Nbr d'éléments non nuls")

#fig1.update_layout(
    #xaxis1=dict(range=[0,5], title_text="Time (s)"),
#)

# Plotting fms vs reg value
fig2 = px.line(df_mean_fms,
    x="sp",
    y="fms",
    log_x=True,
    color="algorithm",
    line_dash="algorithm",
    facet_row="xp",
    error_y="fms_08",
    error_y_minus="fms_02",
    )
fig2.update_yaxes(title_text="Factor Match Score")

# Plotting final loss value vs reg value
fig3 = px.line(df_mean_finalerr,
    x="sp",
    y="final_errors",
    log_x=True,
    log_y=True,
    color="algorithm",
    line_dash="algorithm",
    facet_row="xp",
    error_y="final_errors_08",
    error_y_minus="final_errors_02",
    )
fig3.add_hline(y=0.5, row=1, annotation_text="solution nulle", line=dict(dash='dot'))
fig3.add_hline(y=0.5, row=0, annotation_text="solution nulle", line=dict(dash='dot'))
fig3.update_yaxes(title_text="Fonction de perte normalisée")
# ---------------------------------------------------------
# XP3 --> no room in paper
if False:
    name2 = "./results/xp3_12_28-03-2023"

    df2 = pd.read_pickle(name2)
    nb_seeds = df2["seed"].max()+1 # get nbseed from data

    # Averaging
    df2_mean_sparsity = df2.groupby(["algorithm","unbalance_init","xp"], as_index=False)["sparsity"].mean()
    df2_mean_fms = df2.groupby(["algorithm","unbalance_init","xp"], as_index=False)["fms"].mean()
    df2_mean_finalerr = df2.groupby(["algorithm","unbalance_init","xp"], as_index=False)["final_errors"].mean()

    # Plotting sparsity vs reg value
    fig4 = px.line(df2_mean_sparsity,
        x="unbalance_init",
        y="sparsity",
        log_x=True,
        color="algorithm",
        line_dash="algorithm",
        facet_row="xp",
        )
    fig4.add_hline(y=0.3*df["rank"][0]*df["dim"][0], row=0, annotation_text="True sparsity", line=dict(dash='dot'))
    fig4.add_hline(y=df["rank"][0]*df["dim"][0], row=1, annotation_text="True sparsity", line=dict(dash='dot'))
    fig4.update_layout(
    xaxis=dict(title_text="Init de-scaling")
    )

    # Plotting fms vs reg value
    fig5 = px.line(df2_mean_fms,
        x="unbalance_init",
        y="fms",
        log_x=True,
        color="algorithm",
        line_dash="algorithm",
        facet_row="xp",
        )
    fig5.update_layout(
    xaxis=dict(title_text="Init de-scaling")
    )

    # Plotting final loss value vs reg value
    fig6 = px.line(df2_mean_finalerr,
        x="unbalance_init",
        y="final_errors",
        log_x=True,
        color="algorithm",
        line_dash="algorithm",
        facet_row="xp",
        )
    fig6.add_hline(y=0.5, row=0, annotation_text="solution nulle", line=dict(dash='dot'))
    fig6.add_hline(y=0.5, row=1, annotation_text="solution nulle", line=dict(dash='dot'))
    fig6.update_layout(
    xaxis=dict(title_text="Init de-scaling")
    )

# Storing
# we save twice because of kaleido+browser bug...
newnames = {'ridge balance':'équilibré', 'ridge no-balance': 'pas équilibré', 'ridge no-balance init++':'pas équilibré, init équilibrée', 'sparse(*) no ridge': 'l1 sans l2', 'unregularized': 'pas régularisé'}
for i, fig in enumerate([fig1,fig2,fig3]):#,fig4,fig5,fig6]):
    #if i>0:
        #fig.update_layout(showlegend=False)

    # More editing here that I don't know how to do in the template
    fig.update_xaxes(title_text="")
    fig.update_xaxes(title_text="Paramètre de régularisation", row=1, col=1)
    fig.update_layout(legend_title_text="Algorithme")
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )


    fig.write_image("results/Figure_"+str(i)+".pdf")
    #fig.write_image("Results/"+name+".pdf")
    fig.show()