import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'chunk' })
export class ChunkPipe implements PipeTransform {
  transform(arr: any[], size: number): any[][] {
    return arr.reduce((acc, cur, i) => {
      const idx = Math.floor(i / size);
      const chunk = acc[idx] || (acc[idx] = []);
      chunk.push(cur);
      return acc;
    }, []);
  }
}
