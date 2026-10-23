/** A bounded Unicode prefix: UTF-16 end offset plus complete code-point count. */
export function characterPrefix(text: string, maximum: number): { end: number; count: number } {
  let end = 0;
  let count = 0;
  for (const character of text) {
    if (count === maximum) break;
    end += character.length;
    count++;
  }
  return { end, count };
}
