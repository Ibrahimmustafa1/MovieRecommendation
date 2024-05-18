import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class ApiService {
tmdb_api = 'https://api.themoviedb.org/3/movie/${x}?api_key=f1aca93e54807386df3f6972a5c33b50'
server_api = ' http://127.0.0.1:5000'

  constructor(private http: HttpClient) { }
  getUserMovie(user:any){
    return this.http.get(`${this.server_api}/rated_movies/${user}`)
  }
    getMovieDetails(tmdbId: number): Observable<any> {
        return this.http.get(`https://api.themoviedb.org/3/movie/${tmdbId}?api_key=f1aca93e54807386df3f6972a5c33b50`);
    }

    getMovieRecommendations(user: number): Observable<any> {
        return this.http.get(`${this.server_api}/predict/${user}`);
    }

    getMovieNames(): Observable<any> {
        return this.http.get(`${this.server_api}/movies`);
    }

    getSimilarMovies(movie_id: any): Observable<any> {
        return this.http.get(`${this.server_api}/similar/${movie_id}`);
    }
}
