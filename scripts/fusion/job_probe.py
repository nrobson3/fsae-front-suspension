"""Read-only probe: Fusion version, open documents, the active product type, and the data projects."""
import adsk.core
import adsk.fusion


def main(args):
    app = adsk.core.Application.get()
    out = {"version": app.version, "user": app.currentUser.displayName if app.currentUser else None, "documents": []}
    for i in range(app.documents.count):
        d = app.documents.item(i)
        info = {"name": d.name, "saved": d.isSaved, "modified": d.isModified}
        try:
            des = adsk.fusion.Design.cast(d.products.itemByProductType("DesignProductType"))
            if des:
                info["design_type"] = "parametric" if des.designType == adsk.fusion.DesignTypes.ParametricDesignType else "direct"
                attr = des.attributes.itemByName("fsae", "doc")
                info["fsae_tag"] = attr.value if attr else None
        except Exception as e:  # non-design docs
            info["err"] = str(e)
        out["documents"].append(info)
    out["active"] = app.activeDocument.name if app.activeDocument else None
    try:
        out["projects"] = [app.data.dataProjects.item(i).name for i in range(app.data.dataProjects.count)]
        out["active_project"] = app.data.activeProject.name
    except Exception as e:
        out["projects_err"] = str(e)
    return out
