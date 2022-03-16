package cse512

object HotzoneUtils {
  def parseCoordinates(pointString:String): Array[Double] = {
    val strArr = pointString.split(",")
    val dblArr = new Array[Double](strArr.length)
    var count = 0

    for(str <- strArr) {
      val token = str.toDouble
      dblArr.update(count, token)
      count += 1
    }

    dblArr
  }

  def ST_Contains(queryRectangle: String, pointString: String ): Boolean = {
    var st_contains = false

    val rectangle = parseCoordinates(queryRectangle)
    val lowLat = if(rectangle(0) < rectangle(2)) rectangle(0) else rectangle(2)
    val uprLat = if(rectangle(0) > rectangle(2)) rectangle(0) else rectangle(2)
    val lowLon = if(rectangle(1) < rectangle(3)) rectangle(1) else rectangle(3)
    val uprLon = if(rectangle(1) > rectangle(3)) rectangle(1) else rectangle(3)

    val point = parseCoordinates(pointString)

    // Check if point is within the boundaries
    if (point(0) <= uprLat && point(0) >= lowLat &&
      point(1) <= uprLon && point(1) >= lowLon)
      st_contains = true

    st_contains
  }

  // YOU NEED TO CHANGE THIS PART

}
