@main def exec(bin: String, start:Int, stop:Int, fid: String){
  importCpg(bin)
  val id=7
  cpg.all.toJsonPretty |> s"prop/${fid}_all.json"
  try{
    cpg.all.id(id).l(0).asInstanceOf[Method].dotCfg.l(0) |> s"cfg/${fid}_cfg.dot"
    cpg.all.id(id).l(0).asInstanceOf[Method].dotPdg.l(0) |> s"pdg/${fid}_pdg.dot"
    val func = cpg.all.id(id).l(0).asInstanceOf[Method]
    val src = cpg.identifier.lineNumberGt(start).lineNumberLt(stop)
    val sink = func.call//.filter(!_.name.startsWith("<"))

    sink.reachableByFlows(src).p.mkString("\n") |> s"taint/${fid}.ta"
  }catch{
    case e: Exception => println("Couldn't parse that file.")
  }

}
