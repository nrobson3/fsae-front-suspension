"""Fusion job: close ONLY this project's unsaved documents (design attribute fsae/doc = the tag), discarding them.
Fusion is shared with other sessions' client work: never touch a document without our tag.
Run: python scripts/fusion/fusion_client.py scripts/fusion/job_close_mine.py '{}'"""
import adsk.core
import adsk.fusion


def main(args):
    tag = args.get("tag", "fsae-front-suspension")
    app = adsk.core.Application.get()
    closed = []
    for i in range(app.documents.count - 1, -1, -1):
        d = app.documents.item(i)
        try:
            des = adsk.fusion.Design.cast(d.products.itemByProductType("DesignProductType"))
        except Exception:
            continue
        if not des:
            continue
        at = des.attributes.itemByName("fsae", "doc")
        if at and at.value == tag and not d.isSaved:
            closed.append(d.name)
            d.close(False)
    return {"closed": closed, "left": app.documents.count}
